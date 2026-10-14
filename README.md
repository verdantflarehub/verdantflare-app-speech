# VerdantFlare App Speech

自托管对白运行时与 Studio Speech 接入。当前实现边界为内部 IndexTTS2 API；公开 MCP 与中央 Artifact/Project 交接按设计分步接入，不能通过本内部 API 绕过业务权限。

## 1. 设计事实源

- [对白能力设计](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/01-speech-generation.draft.md)。
- [内部运行时契约](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/speech/03-indextts2-runtime.md)。
- [中央 Schema](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/contracts/drafts/speech-runtime-v1.schema.json)。

## 2. 代码和交付

运行时位于 `services/speech-indextts2-api`；模型权重与用户录音不进入源码仓库。编译镜像由 GitHub Actions 在 release 分支交付，版本标签不可覆盖。部署清单与环境事实归设计仓库。

## 3. 轻量验证

安装运行时 pyproject 中的轻量服务依赖及 httpx 后，设置 PYTHONPATH 为 `services/speech-indextts2-api/src`，运行 `python -m unittest discover -s services/speech-indextts2-api/tests -v`。单元测试使用确定性假引擎验证接口和恢复，不加载 GPU，不代表真实音质通过。模型下载、载入和集成验证在 5090 dev 执行。
