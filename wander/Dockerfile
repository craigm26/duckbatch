# Pollen's microduck Spaces' recipe (spaces/vision-demo/Dockerfile in pollen-robotics/microduck),
# trimmed to what this Space needs: no OpenCV, no video decoder.
FROM python:3.12-slim

# `git` because Hugging Face's builder appends a `git config` layer to every Docker Space, and a
# -slim base without git fails that layer after the image has built.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Gradio's server-side rendering wants Node, which this image does not have.
ENV GRADIO_SSR_MODE=false
# Gradio decides it is on a Space from SYSTEM=spaces, which Hugging Face sets for Gradio Spaces and
# not for Docker ones. Without it the sign-in button is a mock that fails at import.
ENV SYSTEM=spaces
WORKDIR $HOME/app

COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . $HOME/app
EXPOSE 7860
CMD ["python", "app.py"]
