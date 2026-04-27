import huggingface_hub

# Monkey-patch: if cached_download is missing, use hf_hub_download instead.
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download