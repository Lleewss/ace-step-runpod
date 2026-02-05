# ACE-Step 1.5 RunPod Serverless Handler
# Base image includes: CUDA 12.8, ACE-Step models (~15GB), acestep module
FROM valyriantech/ace-step-1.5:latest

# Working directory matches base image
WORKDIR /app

# Install RunPod serverless SDK
# Use explicit python3.11 path to match base image setup
RUN /usr/bin/python3.11 -m pip install --no-cache-dir runpod && \
    /usr/bin/python3.11 -c "import runpod; print('runpod installed successfully')"

# Copy handler into the app directory
COPY handler.py /app/handler.py

# Override the default CMD (start.sh runs FastAPI + Gradio)
# Instead, run the RunPod serverless handler using explicit python path
CMD ["/usr/bin/python3.11", "-u", "/app/handler.py"]
