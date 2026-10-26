# 验证状态和由用户执行的检查

2026-10-24 会话影像更新：**79 项 CPU 测试、8 项 Node 测试、68 项 Firefox 检查通过**，18 项 PostgreSQL / Redis 集成测试未运行。新增隔离 SQLite + 实际 CPU NIfTI 测量验证会话 / 用户 / 租户隔离、已发送文件关联、删除 / 过期结果、真实体积进入追问上下文与自然语言重用历史影像创建未确认草稿。浏览器验证目标下拉框移除、自动二维 / 三维结果刷新、保留切片、关闭后不重开、切会话后的延迟响应、提交失败保留附件及成功后移除附件。未运行真实模型或 GPU 推理，详见 [更新记录与截图](../../../docs/screenshots/README.md)。

2026-10-22 六项细化（最新）：**54 项 CPU 测试、8 项 Node 测试、50 项 Firefox 场景通过**，Ruff / Python 编译 / JS 语法检查通过；18 项数据库集成测试未执行。新增回归验证同会话追问携带前文、合法 clarify 转为自然回答、非法路由不执行工具、对话模型白名单及快照、分割选择不能被路由覆盖、三维实际 affine 几何及高噪声网格面数限制、独立服务鉴权/幂等/阶段/失败/导出（使用伪推理子进程）、退出只清理自身临时 CPU 进程组。浏览器验证删除、Logo/版本隐藏、输入框模型选择、四格位置、三维旋转与显示、固定关闭入口。读取用户 checkpoint 的元数据并准备了 fold 4 的 T1 部署目录；未实例化神经网络、未启动业务服务、未进行 GPU/临床推理。新入口和命令见 [T1 说明](../../agent_app/docs/imaging/T1_CHECKPOINT.md)，截图为明确合成数据，见 [细化记录](../../../docs/screenshots/README.md)。

2026-10-22 深色影像工作区更新（最新）：**41 项 CPU 测试、8 项 Node 测试、42 项 Firefox 浏览器场景通过**，Ruff、JS 语法、HTML ID 和静态资源检查通过。新增 NumPy / SciPy / NiBabel / pydicom / Pillow，锁文件解析为 98 个包；CPU 测试使用 `/tmp` 下的相同固定版本影像依赖，没有安装到应用或训练环境。覆盖 NIfTI/DICOM 几何、LPS→RAS、斜采集重采样、解压限制、mask 标签/物理空间/单位、量化、附件校验、独立 HTTP 服务契约、真实绑定模型身份与离线状态、DICOM 日志内容隐藏。42 项浏览器检查加载实际生产前端与 CSP，以明确合成数据验证三视图、叠加/标签开关、结果重新打开、上传失败/移除、IME、重新生成、SSE、停止、偏好、字体和移动端。未连接业务 DB / Redis / vLLM / nnU-Net；18 项数据库集成测试未运行，真实推理和临床影像仍未验收。默认分割服务未配置，三维表面预览未实现。此次涉及后端和依赖，**需要由用户安装并重启 B、C**；数据库结构不变。详见 [影像接入说明](../../agent_app/docs/imaging/README.md) 与 [截图和浏览器结果](../../../docs/screenshots/README.md)。

2026-10-22 正式页面改版：`node --test tests/web_ui.test.cjs` **8 项通过**，JavaScript 语法检查、HTML ID/资源引用检查、Ruff 检查通过。使用真实 Firefox、与业务相同的 CSP、隔离的内存模拟 HTTP 接口完成 **21 项浏览器场景**：登录与历史会话、文字/表格排版、偏好同意保存和删除、字体调整与浏览器持久化、后台任务独立确认、附件上传说明、SSE 断线后重放去重、输出上限提示保留、模型 HTML 惰性显示、停止回答、会话搜索、手机侧栏/设置和退出登录。截图及机器可读结果位于 `design/ui-implemented-2026-10-22/`。测试临时服务与浏览器已清理；未向业务服务写入测试账户、偏好、任务或附件，未请求真实模型生成。只读确认现有 API 的根页面和三个新版静态资源均返回 200，强制刷新即可加载；本轮不涉及数据库迁移或模型配置变更。

2026-10-22 重复生成排查后执行 `pytest -q tests/test_contracts.py tests/test_routing.py`：**22 passed**，Ruff 与 JavaScript 语法检查通过。新增 MockTransport 检查覆盖流式/非流式回答请求携带正确的 ChatML 停止 ID 和重复惩罚、路由不施加回答惩罚、已有 Run 的解码快照不随 Registry 修改、通用端点不接收未配置的 vLLM 扩展。只读业务记录显示最近三个回答均达到 768 Token 上限，保存回答与 SSE 拼接一致；其中样例两条书目各重复八次且检索引用为空。检查本地 tokenizer、训练模板及导出配置确认结束标记存在差异，但没有获取原始生成 token，不能据此断言它是重复的唯一原因。未调用真实模型生成、改写旧消息或重启服务；生成效果由用户新建会话验证。

2026-10-22 路由故障修复执行 `pytest -q tests/test_routing.py tests/test_contracts.py`：**19 passed**，Ruff 检查通过。使用 MockTransport 和合成模型响应，验证真实配置会发送 JSON Schema 路由约束、普通回答不受格式约束、失败重试携带反馈、截断结果被拒绝、身份伪造参数不进入工具、非法决策达到重试上限后澄清、上游错误继续失败。只读查询确认原业务 Run 为 `ROUTE_INVALID`，两次路由各生成 384 Token；没有修改业务数据、请求真实模型生成或重启任何服务。真实问答结果需要用户重启 API 后重新发送问题验证。

后续双卡故障排查已执行 `pytest -q tests/test_process_control.py`：**3 passed**。测试验证拒绝清理调用者进程组、对不响应 TERM 的临时子进程升级为 KILL、保留正常退出码；该专项不启动 vLLM、不加载 CUDA、不运行应用功能或模型评测。当前单卡服务的只读 `/health` 和 `/v1/models` 检查均返回 200。以下段落记录首次交付时的验证状态。

本次仅写代码和做静态检查，没有启动 API/数据库/Redis/Worker/vLLM，没有执行 pytest、功能冒烟、故障注入、压测或模型效果评测，也没有占用 GPU。依赖锁由 pip 的 `--dry-run --ignore-installed` 为 Python 3.11 解析得到，未安装到现有环境；锁住 93 个包的版本，不含 GPU 框架。

静态检查涵盖 Python AST、Ruff、Shell 语法、JavaScript 语法、YAML/JSON/TOML 解析、项目内导入及文件引用。可重复运行：

```bash
cd agent_app
source .venv/bin/activate
ruff check .
python -m compileall -q src scripts tests alembic
node --check src/pharma_agent/web/app.js  # 有 Node 时可检查，不是前端运行依赖
```

## P0 连通与接口冒烟

按根 README 依次启动后执行：

```bash
python -m pharma_agent.cli model-health
python scripts/smoke.py
python scripts/smoke.py --queue
```

脚本会创建独立合成测试团队；不会访问真实患者数据。队列检查需要 CPU Worker 已启动。若已有太多注册尝试触发 IP 限流，等待一个分钟窗口后再运行。

## CPU 单元测试

```bash
pytest -q tests/test_contracts.py
```

涵盖参数白名单、身份伪造拒绝、偏好同意、窗口边界、YAML 变量替换安全、路径穿越、上游错误映射、截断流失败、Token 用量。测试使用 MockTransport，不请求真实模型。

## 独立数据库集成测试

不要把 TEST_DATABASE_URL 指向业务库。必须使用名称以 `_test` 结尾的专用 PostgreSQL 数据库，以及独立 Redis DB。

```bash
# 例：在当前开发 PostgreSQL 实例中额外创建一次专用库；现有库不受影响。
docker compose exec db createdb -U pharma pharma_agent_test

# 用 .env 相同的随机口令替换下面的 YOUR_DB_PASSWORD。
export TEST_DATABASE_URL='postgresql+psycopg://pharma:YOUR_DB_PASSWORD@127.0.0.1:5433/pharma_agent_test'
export TEST_REDIS_URL='redis://127.0.0.1:6380/15'
DATABASE_URL="$TEST_DATABASE_URL" bash scripts/migrate.sh
pytest -q
```

未设置 TEST_DATABASE_URL 时集成测试会 skip；不能把 skip 视为通过。TEST_REDIS_URL 缺失时队列集成测试也会 skip。测试会创建独立合成租户和记录，不清空现有库；推荐在一次性测试环境执行。测试步骤覆盖：

- 不同租户及同租户不同用户不能读会话/SSE/任务/文件/偏好。
- 幂等请求重用资源，改内容被拒绝，同会话并发被数据库阻止。
- SSE 游标回放不重复事件，流中断不影响底层 Run。
- 非法路由澄清、模型/工具失败有 Trace 且释放会话。
- 偏好删除后新会话不再使用；资料引用有授权范围。
- 进程中断的过期 Run 可恢复为失败，重启后历史/Checkpoint 可继续。
- Job 显式确认和 Worker 重复投递保持幂等。

CI 会使用独立 PostgreSQL/Redis 服务运行 Ruff、迁移和 CPU 测试，无需 GPU；本次没有触发该工作流。真实 vLLM 兼容性、多卡 NCCL、医学回答和路由效果仍需你本机验证。
