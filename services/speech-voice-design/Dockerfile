# syntax=docker/dockerfile:1.7
FROM python:3.11.14-slim-bookworm AS build
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.8.15
COPY scripts/prepare_source.py upstream.lock.json /opt/build/
RUN python /opt/build/prepare_source.py /opt/build/upstream.lock.json /opt/voxcpm
WORKDIR /opt/engine
COPY engine/pyproject.toml engine/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --python /usr/local/bin/python3.11 --frozen --no-dev --no-install-project
COPY pyproject.toml /opt/service/pyproject.toml
COPY src /opt/service/src
RUN uv pip install --python /opt/engine/.venv/bin/python --target /opt/runner --no-deps /opt/service

FROM python:3.11.14-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home speech
COPY --from=build /opt/engine /opt/engine
COPY --from=build /opt/runner /opt/runner
COPY --from=build /opt/voxcpm/src /opt/voxcpm/src
COPY --from=build /opt/voxcpm/LICENSE /opt/voxcpm/LICENSE
COPY --from=build /opt/voxcpm/source-receipt.json /opt/service/source-receipt.json
COPY upstream.lock.json /opt/service/upstream.lock.json
ENV PATH="/opt/engine/.venv/bin:$PATH" PYTHONPATH="/opt/runner:/opt/voxcpm/src" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface NUMBA_CACHE_DIR=/tmp/numba NVIDIA_DRIVER_CAPABILITIES=compute,utility
LABEL org.opencontainers.image.version="0.1.1" org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-speech"
WORKDIR /opt/service
USER 10001:10001
ENTRYPOINT ["python", "-m", "voice_design.cli"]
CMD ["--help"]
