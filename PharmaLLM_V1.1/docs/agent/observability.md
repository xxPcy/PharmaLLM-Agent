# 日志、Trace 与故障定位

API 每次请求生成 request_id 和标准格式 trace_id，响应头提供 `X-Request-ID/X-Trace-ID`。创建 Run 时持久化 OTel trace carrier，后台 Graph/模型/工具继续同一条 Trace；Job 确认时传播到 Worker。

日志只记录结构化元数据：事件名、节点、版本、run/job/request/trace ID、租户 ID、用户 ID hash、输入输出 hash、耗时、错误码和白名单 Token 计数。默认不记录邮箱、密码、JWT、API Key、完整提问、模型正文或图片。SQL echo 关闭，LangSmith 自动上传关闭；Trace 导出仅在显式设置 OTLP URL 后启用。异步任务异常会被消费并脱敏，避免默认 traceback 带出 SQL 参数。

```bash
docker compose --profile observability up -d jaeger
# .env 加入/修改后重启 API 与 Worker：
# OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
```

浏览器访问 http://127.0.0.1:16686 ，按页面显示的 trace_id 搜索。主要 spans 为 `http.request → agent.run → graph.* → model.generate/model.stream/tool.invoke`，还有 `rag.search` 和 `worker.mock_report`。数据库审计记录业务操作、拒绝、下载、删除和工具结果，不能由 Agent 更改。

排查顺序：先查 Run 状态和 error_code，再按 trace_id 找失败节点，核对 graph/prompt/tool/model 版本、路由分支和参数 hash，最后核对来源版本和 Chunk ID。外部 Trace 后端不可用时业务仍可运行；日志和数据库状态用于补充定位。当前没有单独的 Prometheus/GPU 监控服务，时延与 Token 数据来自日志、Trace 和 P7 脚本。

开发故障演示：将 `.env` 改为 `MODEL_MODE=fake`、`ENABLE_FAULTS=true` 并重启 API 后运行：

```bash
python scripts/simulate_tool_failure.py
```

它依次检查模拟 Model 503、模型超时、非法路由 JSON、工具超时和错误输出 Schema。生产模式禁止 FakeModel 和故障开关。

导出 FailMin 对接元数据：

```bash
python -m pharma_agent.cli export-failmin run_... --output reports/failmin.json
```

导出内容包含 hash、版本、事件类型和错误 predicate，不含原问题、原回答或外部可重放的写操作；这不是已经完成 FailMin SDK 对接的声明。操作者可以据此在隔离 Mock 环境补充经授权脱敏的复现输入。导出动作会留审计记录。
