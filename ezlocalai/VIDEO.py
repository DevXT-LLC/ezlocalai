"""Select the configured video backend without importing unused diffusion stacks."""

from ezlocalai.VIDEO_UTILS import DEFAULT_VIDEO_MODEL, VideoGenerationOutOfMemory
from ezlocalai.WAN import WanVideo, is_wan_model

import_success = True


def VIDEO(model=DEFAULT_VIDEO_MODEL, **kwargs):
    if is_wan_model(model):
        return WanVideo(model=model, **kwargs)
    if "ltx" in str(model).lower():
        from ezlocalai.LTX_VIDEO import VIDEO as LTXVideo

        return LTXVideo(model=model, **kwargs)
    raise ValueError(f"Unsupported video model: {model}")
