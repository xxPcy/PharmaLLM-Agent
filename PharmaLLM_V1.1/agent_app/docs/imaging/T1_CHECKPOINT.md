# T1 分割权重接入

此仓库发布分割服务适配代码与结构元数据，不发布 `.pth` 权重。维护者已有的权重可以继续保留在原服务器，GitHub 上传不需要移动它。

当前布局：

```text
PharmaLLM_V1.1/model/
  checkpoint_best.pth                   私有权重，部署者提供，被 Git 忽略
  femoralhead-t1/
    model.json                         模型身份、版本、标签
    dataset.json                       单通道、标签定义
    plans.json                         网络和预处理计划
    fold_4/checkpoint_best.pth          本地准备步骤生成的私有链接
```

在 `agent_app` 中，API `.venv` 执行 `python scripts/prepare_segmentation_model.py` 生成元数据和链接，不实例化网络。不同 checkpoint 生成不同 SHA-256 版本，必须同步 `configs/segmentation.yaml`。

在私有 `.env` 中配置独立 nnU-Net 环境的 `NNUNET_PYTHON`、`NNUNET_MODEL_FOLDER`、`SEGMENTATION_BASE_URL` 和初始化生成的随机 `SEGMENTATION_API_KEY`。

当前独立服务协议为根地址 `http://127.0.0.1:8092`，不能附加 `/v1`。接口密钥只在服务器间使用，不给前端。

```bash
# 在 API .venv 中运行；GPU 1 为示例。
python scripts/start_worker.py --seg-gpu 1
```

常驻进程在独立 Python 环境加载网络到指定 GPU，健康检查确认真实 ready 后启动 CPU Worker。实际预测仍检查原始影像/输出 shape、affine、单位、标签和任务身份。

完整步骤见[本地部署](../../../../docs/LOCAL_DEPLOYMENT.md)，选卡、加载失败、鉴权与协议排查见[常驻分割说明](../../../docs/agent/resident_segmentation.md)。
