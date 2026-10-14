# VerdantFlare App Speech

自托管对白运行时与 Studio Speech 接入。Speech MCP 0.1.1 已接通 Studio 五个工具、IndexTTS2 0.1.3 和原生 Artifact 输出；Project 由调用方显式提交固定引用。用户统一通过 Studio 验证身份与权限。

## 1. 设计事实源

- [对白能力设计](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/01-speech-generation.draft.md)。
- [内部运行时契约](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/03-indextts2-runtime.md)。
- [中央 Schema](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/contracts/drafts/speech-runtime-v1.schema.json)。
- [Studio 调用接入](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/04-studio-speech-integration.md)。

## 2. 代码和交付

运行时位于 `services/speech-indextts2-api`，业务服务位于 [speech-mcp-server](services/speech-mcp-server/README.md)；模型权重与用户录音不进入源码仓库。编译镜像由 GitHub Actions 在 release 分支交付，版本标签不可覆盖。部署清单与环境事实归设计仓库。

## 3. 轻量验证

安装运行时 pyproject 中的轻量服务依赖及 httpx 后，设置 PYTHONPATH 为 `services/speech-indextts2-api/src`，运行 `python -m unittest discover -s services/speech-indextts2-api/tests -v`。单元测试使用确定性假引擎验证接口和恢复，不加载 GPU，不代表真实音质通过。模型下载、载入和集成验证在 5090 dev 执行。
