FROM python:3.11.12-slim-bookworm
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.version="0.2.14" org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-speech"
USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["python", "-m", "speech_mcp"]
