"""
RunPod Serverless Handler for ACE-Step 1.5 Music Generation

This handler initializes the ACE-Step models once at startup and processes
music generation requests via RunPod's serverless infrastructure.
"""

import runpod
import uuid
import base64
import os
import glob

# ACE-Step imports (installed via pip in the base image)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import (
    GenerationParams,
    GenerationConfig,
    generate_music,
)
from acestep.gpu_config import get_gpu_config, set_global_gpu_config

# Configuration from environment variables (set in base image Dockerfile)
PROJECT_ROOT = os.getenv("ACESTEP_PROJECT_ROOT", "/app")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.getenv("ACESTEP_OUTPUT_DIR", "/app/outputs")

# Config paths can be full paths or just model names
# Base image uses full paths like /app/checkpoints/acestep-v15-base
_raw_config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-base")
_raw_lm_path = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")

# Extract just the model name (last directory component)
CONFIG_PATH = os.path.basename(_raw_config_path.rstrip("/\\")) if "/" in _raw_config_path or "\\" in _raw_config_path else _raw_config_path
LM_MODEL_PATH = os.path.basename(_raw_lm_path.rstrip("/\\")) if "/" in _raw_lm_path or "\\" in _raw_lm_path else _raw_lm_path

DEVICE = os.getenv("ACESTEP_DEVICE", "cuda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global handlers - initialized once at cold start
dit_handler = None
llm_handler = None
initialized = False


def initialize_models():
    """Initialize ACE-Step models. Called once during cold start."""
    global dit_handler, llm_handler, initialized
    
    if initialized:
        return True
    
    print("[Handler] Initializing ACE-Step models...")
    
    # Detect and configure GPU
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    print(f"[Handler] GPU Memory: {gpu_config.gpu_memory_gb:.2f} GB, Tier: {gpu_config.tier}")
    
    # Initialize DiT handler
    dit_handler = AceStepHandler()
    status_msg, ok = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        device=DEVICE,
        use_flash_attention=True,
        compile_model=False,
        offload_to_cpu=gpu_config.gpu_memory_gb < 16,
        offload_dit_to_cpu=False,
    )
    
    if not ok:
        print(f"[Handler] ERROR: DiT model failed to load: {status_msg}")
        raise RuntimeError(f"DiT initialization failed: {status_msg}")
    
    print(f"[Handler] DiT model loaded: {CONFIG_PATH}")
    
    # Initialize LLM handler (optional but recommended for better quality)
    llm_handler = LLMHandler()
    if gpu_config.init_lm_default:
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=CHECKPOINT_DIR,
            lm_model_path=LM_MODEL_PATH,
            backend="pt",  # Use PyTorch backend for serverless (more compatible)
            device=DEVICE,
            offload_to_cpu=gpu_config.gpu_memory_gb < 16,
            dtype=dit_handler.dtype,
        )
        if lm_ok:
            print(f"[Handler] LLM model loaded: {LM_MODEL_PATH}")
        else:
            print(f"[Handler] Warning: LLM failed to load: {lm_status}")
            llm_handler = None
    else:
        print("[Handler] Skipping LLM initialization (GPU memory insufficient)")
        llm_handler = None
    
    initialized = True
    print("[Handler] All models initialized successfully!")
    return True


def _is_instrumental(lyrics: str) -> bool:
    """Check if lyrics indicate instrumental music."""
    if not lyrics:
        return True
    lyrics_clean = lyrics.strip().lower()
    if not lyrics_clean:
        return True
    return lyrics_clean in ("[inst]", "[instrumental]")


def handler(event):
    """
    RunPod serverless handler for ACE-Step music generation.
    
    Expected input:
    {
        "input": {
            "prompt": "Upbeat indie pop with jangly guitars",  # or "caption"
            "lyrics": "[Verse]\nWalking down the street...",
            "duration": 60,
            "thinking": false,  # Use LLM for audio codes (better quality but slower)
            "batch_size": 1,
            "inference_steps": 8,
            "audio_format": "mp3"
        }
    }
    
    Returns:
    {
        "audio_base64": "<base64 encoded audio>",
        "format": "mp3",
        "generation_info": "...",
        "seed": "..."
    }
    """
    # Ensure models are initialized
    initialize_models()
    
    inp = event.get("input", {})
    
    # Parse input parameters with sensible defaults
    caption = inp.get("prompt") or inp.get("caption", "")
    lyrics = inp.get("lyrics", "")
    duration = float(inp.get("duration", 60))
    thinking = bool(inp.get("thinking", False))
    batch_size = int(inp.get("batch_size", 1))
    inference_steps = int(inp.get("inference_steps", 8))
    audio_format = inp.get("audio_format", "mp3")
    seed = int(inp.get("seed", -1))
    use_random_seed = inp.get("use_random_seed", True)
    
    # Optional advanced parameters
    bpm = inp.get("bpm")
    key_scale = inp.get("key_scale", "")
    time_signature = inp.get("time_signature", "")
    guidance_scale = float(inp.get("guidance_scale", 7.0))
    vocal_language = inp.get("vocal_language", "en")
    
    print(f"[Handler] Generating music: caption='{caption[:50]}...', duration={duration}s")
    
    # Build generation parameters
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=_is_instrumental(lyrics),
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=key_scale,
        timesignature=time_signature,
        duration=duration,
        inference_steps=inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
        thinking=thinking,
        use_cot_metas=True,
        use_cot_caption=bool(llm_handler),
        use_cot_language=bool(llm_handler),
    )
    
    config = GenerationConfig(
        batch_size=batch_size,
        use_random_seed=use_random_seed,
        audio_format=audio_format,
    )
    
    # Generate music
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=OUTPUT_DIR,
        progress=None,
    )
    
    if not result.success:
        error_msg = result.error or result.status_message or "Unknown generation error"
        print(f"[Handler] Generation failed: {error_msg}")
        raise RuntimeError(f"Music generation failed: {error_msg}")
    
    # Get the first generated audio
    if not result.audios:
        raise RuntimeError("No audio files generated")
    
    audio_path = result.audios[0].get("path")
    if not audio_path or not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")
    
    # Read and encode the audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    # Extract metadata
    seed_values = []
    for audio in result.audios:
        audio_params = audio.get("params", {})
        s = audio_params.get("seed")
        if s is not None:
            seed_values.append(str(s))
    seed_value = ",".join(seed_values) if seed_values else ""
    
    # Clean up generated files
    for audio in result.audios:
        try:
            path = audio.get("path")
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    
    print(f"[Handler] Generation complete, audio size: {len(audio_bytes)} bytes")
    
    return {
        "audio_base64": audio_b64,
        "format": audio_format,
        "seed": seed_value,
        "status": "success",
    }


# Initialize models at import time (during cold start)
# This happens before the first request
try:
    initialize_models()
except Exception as e:
    print(f"[Handler] Warning: Pre-initialization failed: {e}")
    print("[Handler] Models will be initialized on first request")

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
