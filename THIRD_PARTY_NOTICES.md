# 代码与第三方材料

仓库原创代码沿用 V1.0 已有的 MIT 许可证。MIT 许可不替代第三方软件、数据、预训练模型、权重或商标的许可。

- 依赖通过 `pyproject.toml` / `requirements.lock` / `requirements.txt` 安装，仓库未打包第三方 Python 环境。
- 语言模型基于 Qwen3，预训练模型与微调权重未包含在 Git 仓库中。使用或分发时核对对应模型卡、许可与使用条件。
- 分割适配使用独立 nnU-Net 环境；仓库不包含第三方 nnU-Net 源码副本、训练影像或 `.pth` 权重。
- 分割目录中的 `model.json`、`plans.json`、`dataset.json` 是部署结构元数据，保留原标签及计划定义，不代表影像数据或权重的分发授权。元数据中的空 licence 字段不能视作授权。
- HuatuoGPT、中医数据、CMB-Exam、MedQA-ZH 的原始数据、处理后数据和逐题预测均未打包。数据来源和许可需由使用者分别核验。
- `docs/screenshots/` 使用已有隔离浏览器测试中的合成体模与假数据，没有真实患者影像。
- PharmaLLM 标志为项目品牌资产；第三方产品名称仅用于说明技术集成，不表示官方背书。

技术链接：

- [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [vLLM](https://github.com/vllm-project/vllm)
- [LangGraph](https://github.com/langchain-ai/langgraph)
