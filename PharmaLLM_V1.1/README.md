# PharmaLLM Agent 应用

V1.1 是独立的医学对话与影像任务应用。使用相邻 V1.0 训练工程提供的模型，通过本机 vLLM HTTP 接口推理，不要求把训练依赖装进 API 环境。

- [项目首页](../README.md)
- [本地安装、CPU 演示与真实模型启动](../docs/LOCAL_DEPLOYMENT.md)
- [系统与实验说明](../docs/PROJECT_OVERVIEW.md)
- [架构](docs/agent/architecture.md)
- [API 契约](docs/agent/api_contract.md)
- [科室、成员、执行模式与复核](docs/agent/hospital_access.md)
- [分割常驻与选卡](docs/agent/resident_segmentation.md)
- [分割权重接入](agent_app/docs/imaging/T1_CHECKPOINT.md)

发布副本默认使用应用 8082、语言模型 8002、分割 8092，所有服务默认只绑定回环地址。运行前用 `scripts/init_env.py` 生成自己的配置；仓库不含维护者的 `.env`、运行数据、Python 环境或模型权重。
