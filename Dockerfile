# Qwen3-TTS needs a recent PyTorch; cu128 covers Volta through Blackwell (RTX 50xx, sm_120).
# No CUDA devel image needed: nothing is compiled at build or run time.
FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG VERSION=0.0.0.dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# git: sentence-stream is installed from a pinned commit
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /tmp/src/
COPY wyoming_qwen3_tts /tmp/src/wyoming_qwen3_tts/
# Install torch + torchaudio from the CUDA index first, so PyPI's default (newer CUDA) build is not picked
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchaudio \
    && pip install --no-cache-dir /tmp/src/ \
    && rm -rf /tmp/src

VOLUME /data

ENV QWEN3_ASSETS=/data
ENV HF_HOME=/data/hf

EXPOSE 10200

ENTRYPOINT ["python", "-m", "wyoming_qwen3_tts"]
