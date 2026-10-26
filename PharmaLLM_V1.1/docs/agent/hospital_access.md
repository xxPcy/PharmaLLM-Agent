# 执行模式、团队登录与科室复核 · 2026-10-25

本次升级保留现有品牌、字体、主题和四视图，增加执行模式与最小组织权限。应用图版本为 `agent-graph-1.6.0`，业务数据库需要从 `0001` 升级到 `0002`。没有新增运行依赖或修改模型权重。

## 1. “图执行 24 步”是什么

“图”是用节点和连线表示的处理流程。当前主要串行执行，可以把一步理解为执行一个流程节点：

```text
1 准备上下文
2 规划下一步
3 校验工具与参数
4 执行工具
5 读取工具结果
6 生成回答
7 保存并结束
```

若第 5 步后继续调用工具，会再次经过“规划、校验、执行、观察”，所以多一轮通常增加约 4 步。普通直答约 4 个业务节点，单次工具约 7 个，三次工具约 15 个；实际预算由 LangGraph 调度计算。框架限制的是图的执行 superstep，未来若加入并行分支，同一步可以调度多个节点。

`recursion_limit=24` 用来拦住异常循环。它不是对话条数、切片张数或允许 24 次工具调用。`max_tool_calls=3` 是另外的工具数量限制；路由节点内部重试也不等于重新执行完整工具循环。

配置：[graph.yaml](../../agent_app/configs/graph.yaml)。实现：[graph.py](../../agent_app/src/pharma_agent/graph.py)。

## 2. 两种执行模式

输入框的模型选择旁新增执行模式，设置中的“任务执行”也可以修改，两处同步。模式保存在当前账户，默认 **Ask for approval**；每条消息接收时会固定该轮模式并纳入幂等内容校验。

| 模式 | 提出明确分割需求后的行为 |
| --- | --- |
| Ask for approval · 执行前确认 | 创建待确认任务，用户点击任务卡后排队 |
| Full access · 自动执行 | 通过权限、输入和模型契约校验后，由后端自动授权并排队 |

Full access 的交互：上传 `.nii.gz` → 直接发送“帮我分割股骨头和髋臼” → 右侧自动展开源影像 → 任务完成后更新二维 mask 与三维结果。上传成功本身仍只附加文件，不会无请求地开始计算；文件发送成功后从输入框移除，历史消息保留入口。

Full access 仅免除受支持分割任务的逐次执行确认，不给用户增加权限，也不允许模型执行任意代码、调用未登记模型或读取他人文件。工程 Mock 演示任务仍需确认；失败任务不自动无限重试。执行模式不是模型工具参数，模型不能在输出中自行设置它。

已提交任务不会因切换模式而自动取消。执行授权会留下 `job.auto_confirm` 审计；人工确认留下 `job.confirm`。已有 `confirmed_at` 表示获得执行授权，在自动模式下不表示发生过人工按钮点击。

桌面影像工作区位于对话左侧；手机仍使用影像/对话切换。保留手动查看、关闭和下载入口，完成后通常无需再点“打开结果”。

实现：[runs.py](../../agent_app/src/pharma_agent/runs.py)、[tools.py](../../agent_app/src/pharma_agent/tools.py)、[jobs.py](../../agent_app/src/pharma_agent/jobs.py)、[imaging.js](../../agent_app/src/pharma_agent/web/imaging.js)。

## 3. 团队登录代号

原 `tenant_...` 是内部主键，原先注册时自动生成，不适合记忆。本次保留该主键及历史资源，增加独立的可读登录代号。

- 新注册时填写团队名称与代号，例如“骨科研究组” / `guke`。代号为 3–32 个英文字母、数字或连字符，支持 `2026` 这样的纯数字，首字符为字母或数字；自动转为小写，必须全局唯一。
- 留空代号时由服务器生成短代号；建议在注册时自行选择。
- 注册者成为管理员，注册总是新建独立团队，不能靠填写已有代号加入别人团队。
- 管理员创建科室、为医生和审核员建立账户，成员使用同一代号、各自邮箱和密码登录。
- 已有用户迁移后仍可用原长 ID 登录，再到“设置 → 科室与成员”修改团队名称及登录代号。
- 迁移会为旧团队分配 `team-000001` 这类代号，原主键、邮箱、密码及文件归属不变；已有 owner 保持管理员，member 转为 doctor。
- 修改代号不改变内部团队 ID；成员下次登录使用新代号。旧的长 ID 登录继续兼容。

新注册和管理员创建成员的密码为 6–12 个字符，支持纯数字；原较长密码仍可用于登录，已有密码不会被改写。

团队代号不是密码，也不是邀请凭证。当前使用管理员建号及设置初始密码，尚未实现邮件邀请、忘记密码或强制首登改密。

## 4. 最小角色与数据边界

| 操作 | 管理员 owner | 医生 doctor | 审核员 reviewer |
| --- | --- | --- | --- |
| 个人对话 | 可以 | 可以 | 可以 |
| 使用自己有权限的影像提出分割 | 可以 | 可以 | 不可以 |
| 选择 Full access | 可以 | 可以 | 不可以 |
| 管理团队代号、科室、成员及停用账户 | 可以 | 不可以 | 不可以 |
| 导入团队参考资料 | 可以 | 不可以 | 不可以 |
| 提交自己的已完成分割结果复核 | 分配科室后可以 | 分配科室后可以 | 不可以 |
| 浏览复核记录与对应源影像、mask | 本团队已提交结果 | 自己提交的结果 | 本科室已提交结果 |
| 填写通过/退回意见 | 可以，但不能审核自己提交的结果 | 不可以 | 本科室且非本人提交的结果 |
| 查看其他成员全部私有会话/附件 | 不会因角色自动开放 | 不可以 | 不可以 |

首次注册的管理员可以先创建“骨科”，再在成员列表中把自己的账户分配到该科室。管理员不必先绑定科室才能聊天或分割，但提交科室复核需要科室。

科室是组织归属，不代表默认共享全部病例。医生必须在完成的任务卡点击“提交科室复核”，才建立明确共享的复核记录；记录固定源文件、结果文件和提交科室。审核员通过专用只读接口查看这两份资产，不能由此读取其他病例、修改 mask、删除医生文件或访问其会话。

复核支持 pending、approved、changes_requested。退回必须填写意见，已决定的记录不可随意覆盖，不允许自审；原始 mask 不因复核改变。修改后应产生新的分割任务并重新提交。复核意见是项目工作流记录，不等同于已实现医疗文书签发系统。

角色、科室及账户有效状态从数据库读取，Token 不携带固定角色。停用账户或调整审核员科室后，下一次请求使用新权限。模式更改和角色调整不承诺中断已经提交的 GPU 工作；本次没有新增任务取消协议。

实现：[security.py](../../agent_app/src/pharma_agent/security.py)、[hospital.py](../../agent_app/src/pharma_agent/hospital.py)、[hospital.js](../../agent_app/src/pharma_agent/web/hospital.js)。

## 5. 由操作者执行的升级步骤

本次没有对业务数据库执行迁移，也没有替你重启服务或调用真实模型。先等待现有回答与任务结束，在终端 B、C 停止 API 和 Worker，完成备份后升级。

```bash
cd /path/to/PharmaLLM/PharmaLLM_V1.1/agent_app
source .venv/bin/activate
```

可用的备份示例：将文件放在应用构建目录以外；资产目录示例采用当前默认 `var/assets`，自定义过 STORAGE_ROOT 时改为实际目录。

```bash
umask 077
pharma_backup_dir="../backups/access-2026-10-25"
mkdir -p "$pharma_backup_dir"
docker compose exec -T db pg_dump -U pharma -d pharma_agent -Fc > "$pharma_backup_dir/database.dump"
tar -C var -czf "$pharma_backup_dir/assets.tar.gz" assets
```

不删除数据库或数据卷，执行增量迁移：

```bash
bash scripts/migrate.sh
```

重开终端 B：

```bash
cd /path/to/PharmaLLM/PharmaLLM_V1.1/agent_app
source .venv/bin/activate
APP_PORT=8082 python scripts/start_api.py
```

重开终端 C：

```bash
cd /path/to/PharmaLLM/PharmaLLM_V1.1/agent_app
source .venv/bin/activate
python scripts/start_worker.py
```

浏览器 Ctrl+F5，用原团队 ID 登录一次，设置可读代号、科室和成员。vLLM、数据库、Redis 和独立分割服务无需为了本次界面与权限变更重新加载；服务必须仍然可用。无需安装新增依赖或重新分割已完成影像。

迁移文件：[0002_hospital_access.py](../../agent_app/alembic/versions/0002_hospital_access.py)。应用启动要求数据库版本为 `0002`，如果还没迁移，会明确提示运行迁移脚本。迁移前应结束旧 Run，新的图/工具配置不会自动继续执行旧版本排队任务。

## 6. 新接口

| 接口 | 用途 |
| --- | --- |
| `POST /auth/register`，增加 tenant_code | 自定义团队登录代号，新团队管理员 |
| `POST /auth/login`，tenant_id 字段兼容代号或旧 ID | 兼容原客户端；代号忽略大小写 |
| `GET /users/me` | 账户、代号、角色、科室、执行模式和权限 |
| `PATCH /users/me/execution-mode` | 当前账户的默认执行模式 |
| `POST /conversations/{id}/runs`，增加 execution_mode | 固定该轮授权模式并纳入幂等校验 |
| `GET/PATCH /organization` | 管理员查看组织或修改团队信息 |
| `POST /organization/departments` | 管理员创建科室 |
| `POST /organization/members` | 管理员创建医生/审核员 |
| `PATCH /organization/members/{id}` | 管理员更新姓名、角色、科室、状态；保留至少一名有效管理员 |
| `POST /jobs/{id}/review` | 提交本人已完成的分割结果；重复提交复用记录 |
| `GET /reviews` | 按当前角色、团队和科室过滤复核列表 |
| `POST /reviews/{id}/decision` | 授权审核员或管理员填写决定，禁止自审 |
| `GET /reviews/{id}/assets/{asset_id}/metadata,slice,surface,download` | 分别提供元数据、切片、三维和下载的只读接口 |

所有路径前缀为 `/api/v1`。复核资产接口会重新检查记录范围、源结果匹配、文件所有者、状态和有效期；不会放宽原有 `/assets` 所有权检查。

## 7. 验证边界

CPU 用真实 FastAPI 路由、JWT 验证、隔离 SQLite 外键和事务、真实合成 NIfTI 检查权限与策略；另以真实 LangGraph 和受控路由响应验证 Full access 从 Run 快照进入工具、只写一条未发布 Outbox，不执行推理。

SQLite 不验证 PostgreSQL 的行锁、并发约束、迁移实际执行或 Redis 发布；18 项既有 PostgreSQL/Redis 集成测试仍需独立环境运行。迁移只生成并检查了离线 PostgreSQL SQL，没有连接业务库。

浏览器加载当前实际前端与同源 CSP，使用合成 API 数据验证模式、组织管理、独立登录角色、只读复核和自动四视图。截图全部是合成数据，见 [截图与验证汇总](../../../docs/screenshots/README.md)。真实 SFT 意图识别、实际 GPU 推理、运行时延与真实部署需在你升级后验收。

本轮最终验证：**93 项 CPU、8 项 Node、88 项 Firefox 检查通过**；Ruff、JS 语法、HTML ID/资源引用及离线迁移 SQL 检查通过。未运行 18 项 PostgreSQL/Redis 集成测试。

2026-10-26：影像区已移回右侧，终端 C 支持指定 GPU 与常驻服务，最新命令见 [常驻分割说明](resident_segmentation.md)。
