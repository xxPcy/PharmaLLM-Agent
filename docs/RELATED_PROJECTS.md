# PharmaLLM 与 PharmaLLM-Agent

| 仓库 | 定位 |
| --- | --- |
| [PharmaLLM](https://github.com/xxPcy/PharmaLLM) | 医学模型数据处理、微调、离线评测与 vLLM 推理工程 |
| PharmaLLM-Agent | 基于该模型工程的医学助手与 MRI 分割工作台，增加 Agent、异步任务、影像交互和科室复核 |

新仓库名称为 `PharmaLLM-Agent`，在同一账号创建后的地址为 `https://github.com/xxPcy/PharmaLLM-Agent`。产品 Logo 和界面品牌保留 PharmaLLM。版本号继续通过代码目录、发布版本与文档管理，不放入仓库名。

## 给原仓库 README 的引导文案

待新仓库创建并上传成功后，将下面内容放到原仓库 README 的项目介绍后面：

```markdown
## 衍生应用：PharmaLLM-Agent

基于本项目的医疗模型工程，进一步构建了医学对话与 MRI 分割工作台：

- LangGraph Agent 编排、流式对话与会话上下文。
- 自然语言发起股骨头与髋臼分割，四视图查看及体积追问。
- 异步任务、执行确认，以及科室、医生、审核员的结果复核流程。
- 模型在本地 GPU 服务器推理，通过应用入口提供访问。

👉 [查看 PharmaLLM-Agent：代码、界面与部署说明](https://github.com/xxPcy/PharmaLLM-Agent)
```

新仓库首页已添加返回原仓库的链接。线上体验域名仍待配置，接入并验证后再分别补充网址。
