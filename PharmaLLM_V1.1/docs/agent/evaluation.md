# P7 集中评测

**尚未运行，无成功率、P95 或吞吐结果。** 按规格先让 P0–P6 功能路径通过，再执行本页。`tests/evaluation/cases.jsonl` 目前只有四条合成工程样例，不是医学效果评测集。

```bash
cd agent_app
source .venv/bin/activate
python scripts/run_load_test.py \
  --allow-register --concurrency 1,4,8,16 --requests 16 \
  --output reports/evaluation.json
```

该选项会显式创建最多 16 个演示用户，各 Worker 使用不同用户，避免把单用户配额当成整体吞吐。注册默认每 IP 每分钟 20 次，必要时等待下一窗口；生产禁止临时开放注册时，用 `--tokens-file /受控路径/tokens.json` 提供不同用户的 Token JSON 数组。Token 文件应设为 0600，报告不会包含 Token。基线脚本负载小于完整性能研究；根据最终评测需要增加样本、warmup 和重复次数。

输出为实际测得的请求时延 P50/P95、端到端 TTFT、错误率、请求吞吐、服务返回的 Token 用量、工具名称匹配率、Trace ID 和 GPU 前后快照。FakeModel 无真实 Token 计数时写 null；GPU 快照不是峰值监控。工具匹配不代表参数准确、资料支持率或临床正确。

完整效果评测还需人工标注的版本化样本和独立审阅：路由动作/工具/参数，知识问答证据支持，引用正确性，授权边界，多轮承接，真实模型生成质量。可扩展 JSONL 样例并另写评分器；当前不提供未经验证的医学自动裁判。记录模型 revision/hash、Prompt、工具、索引、上下文、输入输出长度、8×A800 的实际分配和其他进程负载。

最终报告应保留原始 JSON、配置、样本版本、失败类别和 baseline；性能变化必须在同负载条件下比较。DPO/GRPO/RLHF、独立 Router 对照和 Reranker 等模型路线不在本次自动执行范围内。
