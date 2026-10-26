# 数据库、迁移与内容保留

业务 schema 由冻结的 `0001_initial.py` 及增量 `0002_hospital_access.py` 管理，应用启动检查 `0002`，不运行 `create_all()`。LangGraph 表使用其官方 `AsyncPostgresSaver.setup()`，由 `scripts/migrate.sh` 的第二步显式初始化和升级。

2026-10-25 增量增加团队登录代号、`departments`、用户科室/显示名/执行模式，以及 `segmentation_reviews`。不改变旧主键和资源归属；旧 member 转为 doctor，owner 不变。科室复核只共享明确提交的源影像与 mask，不共享医生全部私有会话。迁移前等待任务结束、停止 API/Worker 并备份，详见 [升级说明](hospital_access.md)。

业务表：`tenants/departments/users/conversations/messages/agent_runs/run_events/user_memories/assets/tool_jobs/task_outbox/segmentation_reviews/documents/document_chunks/audit_events/user_feedback`。关系、唯一索引和外键见迁移；`audit_events` 的数据库 Trigger 拒绝 UPDATE/DELETE。

- 会话查询过滤租户、owner、软删除状态；thread ID 不能代替授权。
- Run 按租户/用户/会话/幂等键唯一，活动状态有会话唯一约束。
- 偏好按租户/用户/kind 唯一，只接受授权枚举，有 consent_at、expires_at、version。
- 文档按来源/版本唯一，公共文档另有 NULL 租户的唯一索引；内容 hash 防止覆盖历史引用。
- Chunk 按文档/分块/index_version 唯一。无固定向量维度，精确查询前过滤 Embedding 版本；当前没有 ANN 索引。
- Outbox 与 Job 同事务落库；Redis 只承担限流和队列。

## 开发环境保留策略

会话原文、模型回答、工具观察与 SSE delta 会持久化到 PostgreSQL/Checkpoint；附件保存在受控本地目录。它们不是普通日志，但也**没有实现应用层字段加密**。只使用合成或许可允许的公开数据。正式环境应配置磁盘/备份加密、独立数据库角色、访问控制和明确的自动清理周期。

当前默认持久保留直到显式删除；没有暗中清理用户数据的定时器。清理方式：

1. `DELETE /api/v1/memories/{id}` 删除本人偏好。下一次 Run 会重新读取有效偏好，不沿用 Checkpoint 中的旧工具状态。若原会话曾明确展示偏好内容，该历史会话仍按会话策略保留。
2. `DELETE /api/v1/assets/{id}` 删除本人文件正文并写 tombstone；文件下载随后返回 404。任务结果中的资产 ID 可保留，但不再能下载已删文件。
3. `DELETE /api/v1/conversations/{id}` 软删除无活动 Run 的会话，隐藏历史与相关 Run 读取。需要删除正文和 Checkpoint 时，由操作者运行：

```bash
python -m pharma_agent.cli purge-conversation --tenant-id tenant_... --conversation-id conv_...
```

CLI 只接受指定租户内已经软删除的非活动会话，调用官方 `adelete_thread()` 并清理消息、Run、事件和反馈。审计保留无正文元数据。偏好和独立任务/资产有各自生命周期。

业务删除与文件/Checkpoint 属于不同存储操作，清理命令可以重试；若数据库在文件删除后故障，下载会返回 404，需按运维记录核对 tombstone。备份中的历史内容还需纳入单独的备份保留政策。不要对含真实隐私数据的库执行临时演示脚本。
