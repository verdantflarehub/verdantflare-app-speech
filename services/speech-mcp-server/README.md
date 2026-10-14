# Speech MCP Server

## 1. Source and contract

Studio-only speech tools, PostgreSQL task recovery and native Artifact output delivery. Product and deployment facts live in the [central Speech design](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/04-studio-speech-integration.md). `src/speech_mcp/contract.json` projects the central `speech-mcp-v1.schema.json`; update both together.

## 2. Runtime

Python 3.11, HTTP MCP on 8000 and separate HTTPS producer authority on 8001. Startup applies additive database DDL with an advisory transaction lock. The service requires its own database/role, internal credentials, TLS key pair and trusted platform CA. It never loads GPU weights, accepts arbitrary source URLs, forwards user bearers or writes S3 directly.

Configure `SPEECH_DATABASE_URL`, `SPEECH_MCP_TOKEN`, `SPEECH_AUTHORITY_TOKEN`, `SPEECH_PROJECT_URL/TOKEN`, `SPEECH_ARTIFACT_URL/TOKEN`, `SPEECH_RUNTIME_URL/TOKEN`, `SPEECH_PLATFORM_CA_FILE`, `SPEECH_TLS_CERT_FILE`, `SPEECH_TLS_KEY_FILE`, `SPEECH_MODEL_MANIFEST_SHA256`, `ETCD_ENDPOINTS` and `MCP_SERVICE_ENDPOINT` through controlled deployment configuration.

## 3. Verification

Lightweight unit tests use `PYTHONPATH=src python -m unittest discover -s tests -v`. PostgreSQL tests require `SPEECH_TEST_DATABASE_URL` on 5090 dev and create/drop only a uniquely named test schema. No local container or service is started. Images are built only from `release` through the dedicated Actions workflow.
