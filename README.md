# ACE-Step RunPod Serverless

RunPod serverless endpoint for ACE-Step 1.5 music generation.

## Files

- `Dockerfile` - Builds on top of `valyriantech/ace-step-1.5:latest` and adds RunPod handler
- `handler.py` - RunPod serverless handler using the official ACE-Step inference API

## How It Works

The handler uses the official ACE-Step inference API:
- `acestep.handler.AceStepHandler` - DiT (Diffusion Transformer) model handler
- `acestep.llm_inference.LLMHandler` - Language Model for audio codes and metadata
- `acestep.inference.generate_music()` - Unified generation function

Models are initialized once during cold start and reused for subsequent requests.

## Setup on RunPod

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit - ACE-Step serverless handler"
git push origin main
```

### 2. Build Image in RunPod

1. Go to **Serverless → Custom Images**
2. Click **New Custom Image**
3. Configure:
   - **Source**: GitHub
   - **Repo**: `https://github.com/Lleewss/ace-step-runpod.git`
   - **Branch**: `main`
   - **Dockerfile path**: `Dockerfile`
4. Click **Build Image** (takes 5-10 minutes)

### 3. Create Serverless Endpoint

1. Go to **Serverless → Endpoints → New Endpoint**
2. Configure:
   - **Container Image**: Select your newly built image
   - **GPU**: A100 40GB minimum (A100 80GB preferred)
   - **Container Disk**: 50 GB
   - **Max Workers**: 1
   - **Timeout**: 900 seconds
   - **Idle Timeout**: 300 seconds
3. Click **Create Endpoint**

## Usage

### Basic Request

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Upbeat indie pop with jangly guitars and energetic vocals",
      "lyrics": "[Verse]\nWalking down the street\nMusic in my feet\n\n[Chorus]\nWe are alive tonight",
      "duration": 60
    }
  }'
```

### Advanced Request

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Smooth jazz with piano and saxophone",
      "lyrics": "[Verse]\nLate night in the city...",
      "duration": 120,
      "thinking": true,
      "bpm": 85,
      "key_scale": "Bb Major",
      "vocal_language": "en",
      "inference_steps": 12,
      "audio_format": "mp3"
    }
  }'
```

### Response

```json
{
  "audio_base64": "//uQxAAAAAANIAAAAAE...",
  "format": "mp3",
  "seed": "12345",
  "status": "success"
}
```

### Decode Audio (Python)

```python
import base64

audio = base64.b64decode(response["audio_base64"])
with open("song.mp3", "wb") as f:
    f.write(audio)
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | "" | Text description of the music style (alias: caption) |
| lyrics | string | "" | Song lyrics with section markers like [Verse], [Chorus] |
| duration | float | 60 | Duration in seconds (10-600) |
| thinking | bool | false | Use LLM to generate audio codes (higher quality, slower) |
| batch_size | int | 1 | Number of variations to generate |
| inference_steps | int | 8 | Diffusion steps (turbo: 1-20, base: 32-64) |
| audio_format | string | "mp3" | Output format: mp3, wav, flac |
| seed | int | -1 | Random seed (-1 for random) |
| use_random_seed | bool | true | Whether to use random seed |
| bpm | int | null | Tempo in beats per minute (30-300) |
| key_scale | string | "" | Key/scale (e.g., "C Major", "Am") |
| time_signature | string | "" | Time signature (2, 3, 4, 6) |
| vocal_language | string | "en" | Language code for lyrics |
| guidance_scale | float | 7.0 | Prompt guidance strength |

## Notes

- **Cold start**: First request takes 3-6 minutes for image pull + model load
- **Warm requests**: Subsequent requests are much faster (~10-60s depending on duration)
- **Models loaded**: DiT model (~15GB) + optional LLM model (~3GB)
- **GPU requirement**: Minimum 32GB VRAM (A100 40GB or 80GB recommended)
- **Instrumental**: Use `[instrumental]` or `[inst]` as lyrics for instrumental only

## Environment Variables (Pre-configured in Base Image)

| Variable | Default | Description |
|----------|---------|-------------|
| ACESTEP_PROJECT_ROOT | /app | Project root directory |
| ACESTEP_CONFIG_PATH | acestep-v15-base | DiT model path |
| ACESTEP_LM_MODEL_PATH | acestep-5Hz-lm-1.7B | LLM model path |
| ACESTEP_OUTPUT_DIR | /app/outputs | Generated audio output directory |
| ACESTEP_DEVICE | cuda | Device for inference |
