import os, sys
import huggingface_hub
import huggingface_hub.constants as _hf_constants

# ── huggingface_hub 1.x compatibility ────────────────────────────────────────

# 1. hf_cache_home removed in huggingface_hub >= 0.23
if not hasattr(_hf_constants, 'hf_cache_home'):
    _hf_constants.hf_cache_home = os.path.expanduser(
        os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
    )

# 2. cached_download removed in huggingface_hub 1.0
if not hasattr(huggingface_hub, 'cached_download'):
    def _cached_download(*args, **kwargs):
        try:
            return huggingface_hub.hf_hub_download(*args, **kwargs)
        except Exception:
            return args[0] if args else None
    huggingface_hub.cached_download = _cached_download

# 3. HfFolder removed in huggingface_hub 1.0
if not hasattr(huggingface_hub, 'HfFolder'):
    class _HfFolder:
        @staticmethod
        def get_token():
            try: return huggingface_hub.get_token()
            except Exception: return None
        @staticmethod
        def save_token(token):
            try: huggingface_hub.login(token=token, add_to_git_credential=False)
            except Exception: pass
        @staticmethod
        def delete_token(): pass
    huggingface_hub.HfFolder = _HfFolder

# ── JAX compatibility ─────────────────────────────────────────────────────────

# 4. jax.random.KeyArray removed in newer JAX
try:
    import jax
    if not hasattr(jax.random, 'KeyArray'):
        import numpy as np
        jax.random.KeyArray = np.ndarray
except ImportError:
    pass

# ── transformers 5.x compatibility ───────────────────────────────────────────

# 5. FLAX_WEIGHTS_NAME / TF_WEIGHTS_NAME removed in transformers 5.0
#    diffusers 0.20.2 imports these from transformers.utils
try:
    import transformers.utils as _tu
    if not hasattr(_tu, 'FLAX_WEIGHTS_NAME'):
        _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if not hasattr(_tu, 'TF_WEIGHTS_NAME'):
        _tu.TF_WEIGHTS_NAME = "tf_model.h5"
    if not hasattr(_tu, 'TF2_WEIGHTS_NAME'):
        _tu.TF2_WEIGHTS_NAME = "tf_model.h5"
except ImportError:
    pass