FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; swap for a CUDA base if deploying to GPU.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    flask pillow numpy gunicorn

COPY model/ ./model/
COPY app.py .

# Bake the checkpoint into the image so cold starts don't download 375MB.
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

ENV SAM_CHECKPOINT=/app/sam_vit_b_01ec64.pth
ENV PORT=7860
EXPOSE 7860

# One worker: the model is large and each worker loads its own copy.
# Long timeout: CPU encoding of a 1024px image takes a while.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "300", "app:app"]