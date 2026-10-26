# 分割模型目录

Git 仓库只包含结构元数据：`femoralhead-t1/model.json`、`plans.json` 与 `dataset.json`，不包含训练影像或权重。

当前部署约定：单通道 T1 MRI，nnU-Net `3d_fullres`、fold 4，标签 0 背景、1 股骨头、2 髋臼。标签不区分左右。输入序列为用户声明，系统不会把 T2 自动转换成 T1。

部署者将自己有权使用的匹配 checkpoint 放到本目录的 `checkpoint_best.pth`，再在 `agent_app` 执行 `python scripts/prepare_segmentation_model.py`，生成模型布局与权重链接。权重文件和链接均不提交 Git。使用其他权重需同时更新模型注册版本、标签和兼容配置。

元数据不意味着模型权重或原训练数据已开放许可。详见[接入说明](../agent_app/docs/imaging/T1_CHECKPOINT.md)与[第三方说明](../../THIRD_PARTY_NOTICES.md)。
