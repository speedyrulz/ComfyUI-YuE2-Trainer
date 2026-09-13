"""ComfyUI-YuE2-Trainer: train YuE2 LoRAs inside ComfyUI (acoustic MODEL LoRA + planner CLIP LoRA)."""
try:
    from .nodes import YuE2TrainerExtension
except ImportError:
    # Imported outside ComfyUI (unit tests / CLI): the node layer needs the ComfyUI runtime.
    import importlib.util as _ilu
    if _ilu.find_spec("comfy") is not None:
        raise
    YuE2TrainerExtension = None

WEB_DIRECTORY = None


async def comfy_entrypoint():
    return YuE2TrainerExtension()
