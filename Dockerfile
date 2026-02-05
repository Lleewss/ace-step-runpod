# ACE-Step 1.5 RunPod Serverless Handler
# Base image includes: CUDA 12.8, ACE-Step models (~15GB), acestep module
FROM valyriantech/ace-step-1.5:latest

# Install RunPod serverless SDK
RUN pip install --no-cache-dir runpod

# Copy handler into the app directory
COPY handler.py /app/handler.py

# Working directory matches base image
WORKDIR /app

# Override the default CMD (start.sh runs FastAPI + Gradio)
# Instead, run the RunPod serverless handler
CMD ["python", "-u", "/app/handler.py"]
