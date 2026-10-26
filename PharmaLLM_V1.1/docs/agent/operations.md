# 发布副本的运行与排障

推荐布局：Compose 只运行 PostgreSQL / Redis；宿主机运行独立 API `.venv` 和 CPU Worker；vLLM 与 nnU-Net 使用各自的 GPU Python 环境。本发布副本的 Compose 明确只包含基础设施，没有旧的 `app` 全容器 profile。

默认本机端口：API 8082、vLLM 8002、分割 8092、数据库 5433、Redis 6380。可选 Jaeger 为 16686 / 4318。原来的 Dockerfile 保留为打包基础，未声称已经提供可直接运行真实 GPU 服务的全容器编排。

首次安装、CPU 演示与真实模型启动见[本地部署](../../../docs/LOCAL_DEPLOYMENT.md)。已有系统继续使用原运行目录，不在上传目录重复启动。

## 常见问题

| 现象 | 检查 |
| --- | --- |
| 端口占用 | 先确认端口所属服务；不要按名称批量结束其他人的进程 |
| pip SOCKS 依赖缺失 | 允许直连时使用 `bash scripts/install.sh --direct` |
| 数据库认证失败 | 核对私有配置与已初始化卷的实际口令；改 `.env` 不会改旧数据库密码 |
| 模型无法连接 | 检查独立 vLLM、8002 地址和 served model name |
| 分割连接失败 | 旧的裸 `start_worker.py` 只启动 CPU Worker；使用带 `--seg-gpu` 的启动方式 |
| 分割加载中/未就绪 | 等待常驻 ready；使用 `python scripts/check_segmentation.py` 查看安全错误码 |
| 分割身份不一致 | 同步元数据、注册配置和实际 checkpoint 版本 |
| GPU 显存不足 | 检查其他任务和推理峰值，按分配选卡；常驻不代表显卡独占 |
| SSE 不逐字输出 | 检查代理缓冲、缓存、读超时和 `text/event-stream` 响应头 |
| 追问没有任务结果 | 确认同一会话、任务完成、资产有效且当前用户有权限 |

分割详细错误分类见[常驻服务说明](resident_segmentation.md)。

## 升级与数据

代码升级前备份 PostgreSQL 与私有 assets，并记录代码版本；业务表与官方 Checkpoint 均由 `bash scripts/migrate.sh` 初始化。不要在真实业务库执行测试或随意 downgrade，不通过删除数据卷解决迁移问题。

公网展示只暴露应用 HTTPS 入口。现有代码不包含自动备份恢复系统、HA、自动资源预算或完整医疗生产认证，见[公网接入](../../../docs/PUBLIC_ACCESS.md)。
