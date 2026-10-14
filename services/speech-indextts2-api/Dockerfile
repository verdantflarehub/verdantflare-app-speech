# syntax=docker/dockerfile:1.7
FROM python:3.11.14-slim-bookworm AS build
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libsndfile1 libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.8.15
COPY scripts/prepare_source.py /opt/build/prepare_source.py
COPY upstream.lock.json /opt/build/upstream.lock.json
RUN python /opt/build/prepare_source.py /opt/build/upstream.lock.json /opt/indextts
WORKDIR /opt/indextts
COPY engine-environment.toml pyproject.toml
COPY engine.uv.lock uv.lock
RUN --mount=type=cache,target=/root/.cache/uv uv sync --python /usr/local/bin/python3.11 --frozen --no-dev --no-editable
# Compile language normalization transducers once; these are model-free CPU resources.
RUN .venv/bin/python -c "from indextts.utils.front import TextNormalizer; TextNormalizer().load()"
COPY pyproject.toml /opt/service/pyproject.toml
COPY src /opt/service/src
RUN uv pip install --python /opt/indextts/.venv/bin/python --no-deps /opt/service

FROM python:3.11.14-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home speech
COPY --from=build /opt/indextts /opt/indextts
COPY models/manifest.lock.json /opt/service/models/manifest.lock.json
COPY upstream.lock.json /opt/service/upstream.lock.json
ENV PATH="/opt/indextts/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    SPEECH_MODEL_ROOT=/models SPEECH_TASK_ROOT=/data/runtime SPEECH_MODEL_MANIFEST=/opt/service/models/manifest.lock.json \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/tmp/huggingface \
    NLTK_DATA=/models/nltk_data NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
LABEL org.opencontainers.image.version="0.1.1" org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-speech"
WORKDIR /opt/indextts
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn", "speech_runtime.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
