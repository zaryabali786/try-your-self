import io
import os
from contextlib import nullcontext
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import monkey_patch
import apply_net
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)


# ── Device & dtype ────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BASE_PATH = "yisol/IDM-VTON"

print(f"Device: {DEVICE} | Dtype: {DTYPE}")


# ── Load models once at startup ───────────────────────────────────────────────

unet = UNet2DConditionModel.from_pretrained(BASE_PATH, subfolder="unet", torch_dtype=DTYPE)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(BASE_PATH, subfolder="tokenizer", revision=None, use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(BASE_PATH, subfolder="tokenizer_2", revision=None, use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(BASE_PATH, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(BASE_PATH, subfolder="text_encoder", torch_dtype=DTYPE)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(BASE_PATH, subfolder="text_encoder_2", torch_dtype=DTYPE)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(BASE_PATH, subfolder="image_encoder", torch_dtype=DTYPE)
vae = AutoencoderKL.from_pretrained(BASE_PATH, subfolder="vae", torch_dtype=DTYPE)
unet_encoder = UNet2DConditionModel_ref.from_pretrained(BASE_PATH, subfolder="unet_encoder", torch_dtype=DTYPE)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

for model in [unet_encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
    model.requires_grad_(False)

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

pipe = TryonPipeline.from_pretrained(
    BASE_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=DTYPE,
)
pipe.unet_encoder = unet_encoder


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    grayscale = np.array(Image.fromarray(np.array(pil_image)).convert("L"))
    mask = np.where(grayscale > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(mask)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# ── Core inference ────────────────────────────────────────────────────────────

def run_tryon(
    human_img: Image.Image,
    garm_img: Image.Image,
    garment_desc: str,
    use_auto_mask: bool,
    use_auto_crop: bool,
    denoise_steps: int,
    seed: int,
) -> tuple[Image.Image, Image.Image]:

    openpose_model.preprocessor.body_estimation.model.to(DEVICE)
    pipe.to(DEVICE)
    pipe.unet_encoder.to(DEVICE)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = human_img.convert("RGB")

    left = top = 0
    crop_size = None

    if use_auto_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped.size
        human_img = cropped.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if use_auto_mask:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location("hd", "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        raise HTTPException(status_code=400, detail="Manual mask not supported via API — set use_auto_mask=true")

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        "show",
        "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
        "./ckpt/densepose/model_final_162be9.pkl",
        "dp_segm", "-v", "--opts", "MODEL.DEVICE", DEVICE,
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = Image.fromarray(pose_img[:, :, ::-1]).resize((768, 1024))

    autocast_ctx = torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            prompt = "model is wearing " + garment_desc
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

            prompt_c = ["a photo of " + garment_desc]
            negative_prompt_c = ["monochrome, lowres, bad anatomy, worst quality, low quality"]

            with torch.inference_mode():
                (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                    prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt_c,
                )

            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, DTYPE)
            generator = torch.Generator(DEVICE).manual_seed(seed) if seed >= 0 else None

            images = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, DTYPE),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

    if use_auto_crop and crop_size:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray

    return images[0], mask_gray


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Virtual Try-On API",
    description="Send a human image + garment image → get try-on result",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/tryon")
async def tryon(
    human_image: UploadFile = File(..., description="Photo of the person"),
    garment_image: UploadFile = File(..., description="Photo of the garment"),
    garment_description: str = Form("garment", description="Short text description of the garment"),
    use_auto_mask: bool = Form(True, description="Auto-generate body mask"),
    use_auto_crop: bool = Form(False, description="Auto-crop and resize person image"),
    denoise_steps: int = Form(30, ge=20, le=40, description="Diffusion steps (20-40)"),
    seed: int = Form(42, ge=-1, le=2147483647, description="Random seed (-1 = random)"),
    return_mask: bool = Form(False, description="Also return mask image as second part"),
):
    """
    Generate a virtual try-on image.

    Send as multipart/form-data:
    - human_image  : image file
    - garment_image: image file
    - garment_description: e.g. "Short sleeve white t-shirt"
    - use_auto_mask: true/false
    - use_auto_crop: true/false
    - denoise_steps: 20-40
    - seed: integer
    - return_mask: true/false

    Returns: PNG image bytes (the try-on result)
    """
    try:
        human_pil = bytes_to_pil(await human_image.read())
        garm_pil = bytes_to_pil(await garment_image.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image files")

    try:
        result_img, mask_img = run_tryon(
            human_img=human_pil,
            garm_img=garm_pil,
            garment_desc=garment_description,
            use_auto_mask=use_auto_mask,
            use_auto_crop=use_auto_crop,
            denoise_steps=denoise_steps,
            seed=seed,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=image_to_bytes(result_img, "PNG"),
        media_type="image/png",
        headers={"X-Mask-Available": "true" if return_mask else "false"},
    )


@app.post("/tryon/both")
async def tryon_both(
    human_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_description: str = Form("garment"),
    use_auto_mask: bool = Form(True),
    use_auto_crop: bool = Form(False),
    denoise_steps: int = Form(30, ge=20, le=40),
    seed: int = Form(42, ge=-1, le=2147483647),
):
    """
    Same as /tryon but returns JSON with both images as base64.

    Returns:
    {
        "result_image": "<base64 PNG>",
        "mask_image":   "<base64 PNG>"
    }
    """
    import base64

    try:
        human_pil = bytes_to_pil(await human_image.read())
        garm_pil = bytes_to_pil(await garment_image.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image files")

    try:
        result_img, mask_img = run_tryon(
            human_img=human_pil,
            garm_img=garm_pil,
            garment_desc=garment_description,
            use_auto_mask=use_auto_mask,
            use_auto_crop=use_auto_crop,
            denoise_steps=denoise_steps,
            seed=seed,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "result_image": base64.b64encode(image_to_bytes(result_img)).decode(),
        "mask_image": base64.b64encode(image_to_bytes(mask_img)).decode(),
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
