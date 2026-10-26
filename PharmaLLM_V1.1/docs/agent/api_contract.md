# API 契约

完整 Schema 由运行后的 `/docs` 和 `/openapi.json` 提供。完整应用所有 `/api/v1` 业务接口都需要 Bearer Token，注册和登录除外。列表有分页/条数上限；不属于当前用户/租户的资源统一 404。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health/live`、`/health/ready` | 进程状态、DB/Redis/所配置模型检查 |
| POST | `/api/v1/auth/register` | email、6–12 字符 password、可选 tenant_name；返回 Token 和团队 ID |
| POST | `/api/v1/auth/login` | tenant_id、email、password |
| GET | `/api/v1/users/me` | 当前认证身份 |
| POST / GET | `/api/v1/conversations` | 创建、分页列出本人会话 |
| DELETE | `/api/v1/conversations/{id}` | 无活动 Run 时软删除 |
| GET | `/api/v1/conversations/{id}/messages` | 最近消息，支持 limit/before |
| POST | `/api/v1/conversations/{id}/runs` | message、可选 stream / asset_ids / model_alias / segmentation_model_id；返回 202 和 Run 描述 |
| GET | `/api/v1/runs/{id}` | 状态、回答、引用、Token 用量和 trace_id |
| GET | `/api/v1/runs/{id}/events` | SSE，支持 Last-Event-ID 或 after |
| POST | `/api/v1/runs/{id}/cancel` | 请求取消；运行中的 Run 先进入 cancelling |
| GET / POST | `/api/v1/memories` | 当前偏好列表、显式 consent=true 的 upsert |
| DELETE | `/api/v1/memories/{id}` | 删除本人偏好 |
| POST | `/api/v1/jobs/drafts` | title；仅生成 mock_report 草稿 |
| POST | `/api/v1/jobs/{id}/confirm` | confirmed=true；返回 202，队列仍可能待发布 |
| GET | `/api/v1/jobs`、`/api/v1/jobs/{id}` | 本人任务与结果资产 ID |
| POST | `/api/v1/assets` | 兼容单文件上传，走真实文件解析；默认上限 64MiB，可配置 |
| GET / DELETE | `/api/v1/assets/{id}/download` / `/api/v1/assets/{id}` | 授权附件下载、删除文件正文 |
| POST | `/api/v1/documents` | owner 导入本团队文本资料，需来源/版本/许可/日期 |
| GET | `/api/v1/references/{chunk_id}` | 按资料可见范围回源 |
| POST | `/api/v1/feedback` | run_id、rating ±1、category |

Run 和任务草稿创建可带 `Idempotency-Key`，同一个键重试返回同一资源，改内容返回 `IDEMPOTENCY_CONFLICT`。缺失时服务端生成新键；需要网络重试的客户端应自己保留并复用原键。

Run 状态：`queued → running → completed/failed`；取消时为 `queued → cancelled` 或 `running → cancelling → cancelled`。任务状态：`draft → queued → running → succeeded/failed`，失败的 Mock 任务可再次显式确认。

SSE 事件：`status`、`delta`、`tool_start`、`tool_end`、`completed`、`error`。每个持久事件都有递增整数 ID；重连只返回当前 Run 中 ID 大于游标的事件。传输层故障可返回无 ID 的 `EVENT_STREAM_UNAVAILABLE`，客户端应重连，不应创建新 Run。

错误返回：

```json
{"error":{"code":"CONVERSATION_BUSY","message":"当前会话正在回答，请等待完成","retryable":true},"request_id":"req_...","trace_id":"32位十六进制"}
```

代码包括 `AUTH_REQUIRED`、`NOT_FOUND`、`PERMISSION_DENIED`、`INVALID_REQUEST`、`INVALID_TOOL_ARGS`、`MODEL_TIMEOUT`、`MODEL_UNAVAILABLE`、`MODEL_BAD_RESPONSE`、`MODEL_CONTEXT_OVERFLOW`、`TOOL_TIMEOUT`、`TOOL_BAD_RESPONSE`、`RATE_LIMITED`、`RUN_INTERRUPTED`、`QUEUE_UNAVAILABLE` 等。参数错误不会把用户输入、密码或上游原始正文回显进错误详情。


### 影像与模型选择（当前实现）

- `GET /api/v1/model-info`：已绑定的模型、健康状态、可选对话模型别名；不返回模型密钥或 endpoint URL。
- `GET /api/v1/imaging/capabilities`：文件限制、已允许的分割模型和真实服务探测状态。
- `POST /api/v1/imaging/uploads`：multipart `file` 可重复；支持 NIfTI、PNG/JPEG、DICOM 单层/序列/ZIP 及 TXT/JSON；可声明 `modality`、`conversation_id`。
- `GET /api/v1/assets`、`GET /api/v1/assets/{id}/metadata`：本人文件及解析信息。
- `GET /api/v1/assets/{id}/preview`、`GET /api/v1/assets/{id}/slice`：授权图像 / 物理空间切片。
- `GET /api/v1/assets/{id}/surface`：仅对真实 mask 返回有面数限制的物理坐标体素边界网格。
- `POST /api/v1/assets/{id}/modality`：multipart 模态声明，不执行影像序列转换。
- `GET /api/v1/conversations/{id}/imaging-jobs`：本会话最近 50 个分割任务及结果。

`model_alias` 只允许服务端配置的对话模型，固定到本 Run 的回答模型快照，不改变其他会话的模型。当前页面不再发送 `segmentation_model_id`，分割结构由自然语言路由结合服务端配置选择；该可选参数为已有 API 客户端保留。对于显式传入参数的客户端，`segmentation_model_id` 是本轮的明确分割选择；路由工具不能改用另一个模型。模型、附件、模态和任务类型仍需通过服务端校验。幂等哈希涵盖这些选择。

普通追问/纠错使用原会话历史；生产策略在路由不合法时转为 `ROUTE_FALLBACK` 的普通回答，不执行该非法路由的工具动作。错误码保留在运行详情；先前已有的真实工具观察不会被抹去。分割服务的独立 HTTP 协议和 T1 权重入口见 [T1 接入说明](../../agent_app/docs/imaging/T1_CHECKPOINT.md)。

2026-10-24：消息提交被接受后，客户端清除本轮输入附件，历史消息的 `asset_ids` 保持可查看。后续请求可发送空 `asset_ids`：服务端按当前用户 / 租户 / 会话读取已提交影像和最新分割结果，进行状态、保留期限与源结果关系验证，再把有效测量提供给回答节点。没有改变 Run 请求或响应结构。分割完成后浏览器自动刷新当前影像的叠加和三维结果，旧会话异步响应受会话及视图版本保护。
