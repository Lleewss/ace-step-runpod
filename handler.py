"""
RunPod Serverless Handler for ACE-Step 1.5 Music Generation

This handler initializes the ACE-Step models once at startup and processes
music generation requests via RunPod's serverless infrastructure.
"""

import sys
import traceback

print("[Handler] Starting ACE-Step RunPod handler...", flush=True)
print(f"[Handler] Python version: {sys.version}", flush=True)
print(f"[Handler] Python path: {sys.path}", flush=True)

import runpod
import base64
import os

print("[Handler] Basic imports successful", flush=True)

# Lazy imports - will be done in initialize_models()
AceStepHandler = None
LLMHandler = None
GenerationParams = None
GenerationConfig = None
generate_music = None
get_gpu_config = None
set_global_gpu_config = None

# Configuration from environment variables (set in base image Dockerfile)
PROJECT_ROOT = os.getenv("ACESTEP_PROJECT_ROOT", "/app")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.getenv("ACESTEP_OUTPUT_DIR", "/app/outputs")

print(f"[Handler] PROJECT_ROOT: {PROJECT_ROOT}", flush=True)
print(f"[Handler] CHECKPOINT_DIR: {CHECKPOINT_DIR}", flush=True)
print(f"[Handler] OUTPUT_DIR: {OUTPUT_DIR}", flush=True)

# List checkpoint directory
if os.path.exists(CHECKPOINT_DIR):
    print(f"[Handler] Checkpoints found: {os.listdir(CHECKPOINT_DIR)}", flush=True)
else:
    print(f"[Handler] WARNING: Checkpoint dir does not exist!", flush=True)

# Config paths can be full paths or just model names
# Base image uses full paths like /app/checkpoints/acestep-v15-base
_raw_config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-base")
_raw_lm_path = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")

print(f"[Handler] Raw config path: {_raw_config_path}", flush=True)
print(f"[Handler] Raw LM path: {_raw_lm_path}", flush=True)

# Extract just the model name (last directory component)
CONFIG_PATH = os.path.basename(_raw_config_path.rstrip("/\\")) if "/" in _raw_config_path or "\\" in _raw_config_path else _raw_config_path
LM_MODEL_PATH = os.path.basename(_raw_lm_path.rstrip("/\\")) if "/" in _raw_lm_path or "\\" in _raw_lm_path else _raw_lm_path

print(f"[Handler] CONFIG_PATH: {CONFIG_PATH}", flush=True)
print(f"[Handler] LM_MODEL_PATH: {LM_MODEL_PATH}", flush=True)

DEVICE = os.getenv("ACESTEP_DEVICE", "cuda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global handlers - initialized once at cold start
dit_handler = None
llm_handler = None
initialized = False


def initialize_models():
    """Initialize ACE-Step models. Called once during cold start."""
    global dit_handler, llm_handler, initialized
    global AceStepHandler, LLMHandler, GenerationParams, GenerationConfig, generate_music
    global get_gpu_config, set_global_gpu_config
    
    if initialized:
        return True
    
    print("[Handler] Importing ACE-Step modules...", flush=True)
    
    try:
        from acestep.handler import AceStepHandler as _AceStepHandler
        from acestep.llm_inference import LLMHandler as _LLMHandler
        from acestep.inference import (
            GenerationParams as _GenerationParams,
            GenerationConfig as _GenerationConfig,
            generate_music as _generate_music,
        )
        from acestep.gpu_config import get_gpu_config as _get_gpu_config, set_global_gpu_config as _set_global_gpu_config
        
        AceStepHandler = _AceStepHandler
        LLMHandler = _LLMHandler
        GenerationParams = _GenerationParams
        GenerationConfig = _GenerationConfig
        generate_music = _generate_music
        get_gpu_config = _get_gpu_config
        set_global_gpu_config = _set_global_gpu_config
        
        print("[Handler] ACE-Step imports successful!", flush=True)
    except ImportError as e:
        print(f"[Handler] ERROR importing ACE-Step: {e}", flush=True)
        print(f"[Handler] Traceback: {traceback.format_exc()}", flush=True)
        raise
    
    print("[Handler] Initializing ACE-Step models...", flush=True)
    
    # Detect and configure GPU
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    print(f"[Handler] GPU Memory: {gpu_config.gpu_memory_gb:.2f} GB, Tier: {gpu_config.tier}", flush=True)
    
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
        print(f"[Handler] ERROR: DiT model failed to load: {status_msg}", flush=True)
        raise RuntimeError(f"DiT initialization failed: {status_msg}")
    
    print(f"[Handler] DiT model loaded: {CONFIG_PATH}", flush=True)
    
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
            print(f"[Handler] LLM model loaded: {LM_MODEL_PATH}", flush=True)
        else:
            print(f"[Handler] Warning: LLM failed to load: {lm_status}", flush=True)
            llm_handler = None
    else:
        print("[Handler] Skipping LLM initialization (GPU memory insufficient)", flush=True)
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
    try:
        print(f"[Handler] Received request: {event}", flush=True)
        
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
        
        print(f"[Handler] Generating music: caption='{caption[:50]}...', duration={duration}s", flush=True)
        
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
            print(f"[Handler] Generation failed: {error_msg}", flush=True)
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
        
        print(f"[Handler] Generation complete, audio size: {len(audio_bytes)} bytes", flush=True)
        
        return {
            "audio_base64": audio_b64,
            "format": audio_format,
            "seed": seed_value,
            "status": "success",
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"[Handler] ERROR: {error_msg}", flush=True)
        print(f"[Handler] Traceback: {traceback.format_exc()}", flush=True)
        return {"error": error_msg, "status": "failed"}


print("[Handler] Handler module loaded, starting RunPod serverless...", flush=True)

# Start the RunPod serverless handler
# Models will be initialized on first request (not at import time)
runpod.serverless.start({"handler": handler})
