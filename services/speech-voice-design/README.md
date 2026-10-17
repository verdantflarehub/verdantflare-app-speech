# Voice candidate runner

## 1. Scope

Developer runner for pinned VoxCPM2 reference cloning and description-based voice design. It is not an exposed Studio tool. Current deployment and behavior decisions belong to the [central design](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/05-voice-design.md).

Source version `0.1.0`; GPU execution and audio quality are unvalidated. The existing IndexTTS2 service remains the registered speech backend. No model weights or user audio belong in this repository.

## 2. Commands

The image provides `python -m voice_design.cli` with these subcommands:

- `check-request --request request.json`: validate the internal request without inference.
- `prepare-model --model-root /models/voxcpm2`: explicit, resumable hash-locked model download, separate from inference. Existing differing files are refused.
- `model-probe --model-root /models/voxcpm2`: verify the complete local model before loading.
- `gpu-probe`: require one assigned CUDA device, free-memory check and small matrix operation.
- `generate --request request.json --reference source.wav --model-root /models/voxcpm2 --out /data/candidate-v1`: one candidate, fresh output directory required.

Request fields and mode constraints are in [request.schema.json](src/voice_design/request.schema.json). `design` requires reference traits, requested changes and the final target description; the reference's bytes are validated and preserved as source evidence, but are **not passed to the model**. `clone` supplies those bytes to the model and optionally applies a performance instruction. Both modes preserve original text and the actual model input. No transcription is fabricated.

Successful execution produces `request.json`, `source-reference.wav`, `started.json`, `original.wav`, `provenance.json` and `completed.json`. A technically eligible candidate also has `reference.wav` (PCM24). `failed.json` records a sanitized failure without retries. A started directory without completion requires investigation; never rerun into it. Completion means candidate creation, not approval, registration or a verified transcript.

## 3. Checks and build

Install only the lightweight dependencies in `pyproject.toml` for unit tests; run `PYTHONPATH=src python -m unittest discover -s tests -v` from this directory. Tests use a fake engine and synthetic in-memory signals, never import VoxCPM or run inference.

`engine/pyproject.toml` and `engine/uv.lock` pin the Linux CUDA environment. The build fetches a checksum-locked source archive and copies only source plus its license. Upstream training, web UI, denoising and text normalization extras are excluded. Upstream source is not modified. The GitHub Actions workflow builds the image; no local container builds. Actual runtime and GPU checks use 5090 dev according to the central deployment rules.
