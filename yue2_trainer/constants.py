"""Checkpoint-native token ids and prompt strings for YuE2 (mirrors comfy.text_encoders.yue2)."""

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
CONTEXT = 24576
FRAMES_PER_SECOND = 25
LATENT_CHANNELS = 64
SAMPLES_PER_FRAME = 1920
SAMPLE_RATE = 48000

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}

# LoRA key prefixes understood by ComfyUI's loaders (comfy/lora.py).
MODEL_KEY_PREFIX = "diffusion_model."      # matched by model_lora_keys_unet (generic format)
CLIP_KEY_PREFIX = "text_encoders."         # matched by model_lora_keys_clip (generic format)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
