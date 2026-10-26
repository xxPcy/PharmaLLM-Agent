# PharmaLLM 系统与实验说明

## 项目范围

V1.0 实现中文医疗指令数据处理、Qwen3-8B-Base 全参数 SFT 与 LoRA、选择题评测及 vLLM 服务脚本。V1.1 将语言模型接入有权限和任务状态的 Agent 应用，并增加 T1 MRI 分割、四视图浏览、测量追问与科室复核。

发布仓库保留两个目录，便于对应训练产物和应用服务。训练数据与权重不随代码发布；应用也不在启动时自动下载模型。

## 用户流程与场景

| 场景 | 系统行为 | 边界 |
| --- | --- | --- |
| 中文医学知识问答 | 独立模型流式生成，保存会话，可在窗口内追问 | 模型仍会出错，离线选择题准确率不等于开放问答质量 |
| 带出处的资料检索 | 限定租户/公共范围，返回片段与来源元数据 | 默认 hash Embedding 为演示基线，未部署医学语义模型 |
| T1 分割 | 上传影像，自然语言表达支持的部位，按模式确认或自动排队 | 当前注册股骨头/髋臼任务，不支持任意部位和任意模态 |
| 浏览与测量追问 | 三个二维视图、三维 mask、体积、下载，同会话读取有效结果 | 标签不分左右；单位未知时不编造体积 |
| 科室结果复核 | 医生主动提交结果，所属科室审核员查看授权资产并决定 | 不是电子病历签发或完整医院信息系统 |
| CPU 工程演示 | FakeModel、合成参考资料和异步模拟任务 | 不能冒充真实模型、真实病例或真实分割 |

## 运行结构

浏览器与 API 同源。FastAPI 负责身份、会话、资产和任务接口；LangGraph 在应用进程内执行有界状态图，通过模型网关调用独立 vLLM。PostgreSQL 保存业务事实、消息、事件、审计、官方 Checkpoint 和 pgvector 资料。Redis 承担队列与限流，Celery Worker 处理长任务，独立分割服务与 nnU-Net 常驻进程负责 GPU 推理。

API 与 Celery 不加载 Torch 模型。两类模型可以选择不同物理 GPU；分割启动器通过 GPU UUID 固定设备，预加载成功后才接受任务。已有多卡排障记录不代表所有硬件都有同一问题；常驻也不代表显卡独占。

## Agent 的边界

主要路径：`prepare_context → plan_or_route → validate_action → execute_tool → observe → generate_answer → finalize`。

- 图递归步数上限 24，覆盖规划、校验、观察等状态图节点，不是 24 次 GPU 推理或 24 次工具调用。
- 工具循环默认至多 3 次，路由格式失败有限重试；超限停止继续调用工具。
- 模型给出 JSON 动作后，应用校验白名单、输入 Schema、授权、版本与输出；无效动作不直接执行。
- 普通 SFT 模型未声明原生 Tool Calling 训练。结构化输出能力来自服务端约束解码与后端验证。
- 默认保留最多 20 条历史消息，并受 UTF-8 字节预算约束；已接受的同会话文件和有效分割结果单独补充为业务上下文。
- Full access 是后端固定的用户执行策略，只自动授权支持的分割操作；模型不能靠文字自行提权。

## 工程设计

| 设计 | 实现与作用 | 主要代码 |
| --- | --- | --- |
| 租户与用户隔离 | 资源查询绑定 actor，模型不能传入其他身份 | `security.py`、`db.py`、`api.py` |
| 医院最小权限 | 管理员管理成员，医生持有个人病例，审核员仅读主动提交的科室结果 | `hospital.py`、`schemas.py` |
| 幂等 | 请求键、数据库约束、任务领取与版本校验抑制重复执行 | `runs.py`、`jobs.py`、`db.py` |
| 执行确认 | 草稿与授权分离，支持账户级 Ask / Full 模式 | `tools.py`、`jobs.py`、`schemas.py` |
| 队列可靠性 | 授权与 Outbox 同事务；发布有限重试，领取租约与心跳 | `jobs.py`、`runs.py` |
| 断线恢复 | SSE 事件持久化，订阅与运行分离，按事件 ID 重连 | `api.py`、`runs.py`、`web/app.js` |
| 崩溃边界 | Checkpoint、会话锁与租约检测，失败写操作不盲目回放 | `runs.py`、`graph.py` |
| 文件与影像 | 私有资产授权、上传/解压/体素上限、几何与标签校验 | `storage.py`、`imaging.py`、`segmentation.py` |
| 结果追问 | 只读取同会话已授权结果、体素体积和有效单位 | `imaging_context.py`、`graph.py` |
| 审计与观测 | 追加式审计、结构化日志、trace、可选 OTel | `observability.py`、`db.py` |
| GPU 生命周期 | 显式选卡、自有进程组清理、常驻 ready 协议 | `resident_model.py`、`scripts/start_worker.py` |

代码路径相对于 `PharmaLLM_V1.1/agent_app/src/pharma_agent/`；`scripts/` 相对于 `agent_app/`。完整说明见[架构](../PharmaLLM_V1.1/docs/agent/architecture.md)、[数据模型](../PharmaLLM_V1.1/docs/agent/database.md)与[权限流程](../PharmaLLM_V1.1/docs/agent/hospital_access.md)。

## 训练与评测记录

全参数训练使用 4×A800 80GB、BF16、DeepSpeed ZeRO-3、1 个 epoch、4096 序列长度和有效 batch 32；样本为 60000 条通用医学、20000 条中医训练及 2000 条验证数据。全参数 SFT 的 train loss 1.4634、eval loss 1.4515，训练约 37.55 小时。LoRA 是独立对照路径。

离线评测采用 CMB-Exam val 280 题与 MedQA-ZH test 3426 题，温度 0、最多 8 个新 Token、只输出选项字母。SFT 分别为 58.93% 和 81.52%；原始 Base 为 35.71% 和 21.89%；官方后训练 Qwen3-8B 为 67.14% 和 84.71%。结果未超过官方模型。见[历史汇总](../PharmaLLM_V1.0/docs/results/model_evaluation_comparison_qwen3_8b.md)。

已有精确去重实现；近重复、基准污染等治理配置不代表全部环节已经实现并完成验证。`quantize_awq.sh` 仍是占位入口，本次不开源量化权重，也不声明完整可复现的 AWQ 产线。

## 分割几何与测量

当前适配器为单通道 T1、nnU-Net `3d_fullres`、fold 4，标签 0 背景、1 股骨头、2 髋臼。输出需与原始影像 shape、affine、单位及允许标签一致。体积来自原始空间的标签体素计数和 affine 体素体积，不从网页三维简化网格估算。

没有独立公开的 Dice、HD95 或外部临床验证结论。三维视图对显示规模作限制，下载 mask 与原始体积统计保持独立。

## 开源范围与后续上线

本仓库包含源码、依赖声明、迁移、配置模板、测试、结构元数据和合成界面截图。没有真实用户记录、影像、数据库、日志、私钥、运行 `.env`、模型权重或训练/评测原始数据。

现阶段推荐本地 GPU 推理，通过固定 HTTPS 隧道发布单一应用入口，见[公网访问方案](PUBLIC_ACCESS.md)。公开展示仍需完成进程托管、域名配置、访客策略、资源额度、数据清理与端到端网络测试；代码开源本身不会自动完成这些工作。当前未实现 SSO、PACS、完整医院权限、对象存储、Kubernetes、高可用或临床生产认证。
