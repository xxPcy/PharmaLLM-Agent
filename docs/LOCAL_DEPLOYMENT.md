# 本地运行：API、模型与分割分别管理

这份说明面向新部署。已经运行系统的维护者继续使用原开发目录和原配置；无需在上传副本里再次启动服务。同一台机器同时启动两个副本，会因相同端口或 Compose 项目名产生冲突或复用数据服务。

默认端口：应用 8082、vLLM 8002、分割 8092、PostgreSQL 5433、Redis 6380。均只绑定本机回环地址。

## 1. CPU 工程演示

从仓库根目录进入应用。需要 Python 3.11–3.13（建议 3.12）、Docker Compose：

```bash
cd PharmaLLM_V1.1/agent_app
python3 scripts/init_env.py --mode fake
bash scripts/install.sh
source .venv/bin/activate
docker compose up -d --wait db redis
bash scripts/migrate.sh
python -m pharma_agent.cli seed-demo
python scripts/start_api.py
```

另一个终端进入同一应用目录，启动 CPU Worker：

```bash
source .venv/bin/activate
python scripts/start_worker.py
```

浏览器打开 `http://127.0.0.1:8082`。注册新团队会创建管理员，之后可在设置中创建科室和成员。注册是创建独立团队，不是加入任意已有团队。

`init_env.py` 生成私有 `.env`、随机数据库口令、JWT 密钥与分割接口密钥；拒绝覆盖已有配置。数据库首次初始化后的口令不会因重新编辑 `.env` 而自动改变。`seed-demo` 只导入明确标注的合成参考文本，默认 hash 检索不代表医学语义检索。

FakeModel 用于检查页面、状态与工具契约，不能用于医学回答质量或真实分割验证。若 pip 继承了不可用的 SOCKS 代理，可在允许直连的环境执行 `bash scripts/install.sh --direct`；它不会修改系统代理。

## 2. 接入自己的语言模型

保留 API `.venv`，使用独立 vLLM Python 环境。开发机曾使用 vLLM 0.30.0；其他 CUDA / 驱动 / vLLM 组合需单独验证。仓库不打包模型或完整 GPU 环境。

编辑私有 `.env`：

```dotenv
MODEL_MODE=openai
PHARMA_VLLM_BASE_URL=http://127.0.0.1:8002/v1
ROUTER_BASE_URL=http://127.0.0.1:8002/v1
PHARMA_VLLM_MODEL_NAME=pharmallm-v1
ROUTER_MODEL_NAME=pharmallm-v1
MODEL_PATH=/path/to/your/merged-model
VLLM_PORT=8002
```

`MODEL_PATH` 必须是包含模型配置、分词器和完整权重的模型目录；LoRA adapter 目录不能直接替代合并后的模型。当前模型配置中的停止 Token 和约束输出能力针对 Qwen/vLLM，替换服务或分词器时重新核对。

**终端 A：** 激活自己的 vLLM 环境，进入 `agent_app`：

```bash
python scripts/serve_vllm.py --gpus 0 --tp 1
```

GPU 0 只是示例，按实际分配更换。脚本检查显存，不会结束其他进程。先用单卡验证再按实际并发决定是否多卡；八张 A800 不需要全部分给一个 8B 模型。

**终端 B：** 激活 API `.venv`，进入 `agent_app`：

```bash
python scripts/start_api.py
```

## 3. 接入 T1 分割权重

当前适配器约定单通道 T1 MRI、`3d_fullres`、fold 4、背景/股骨头/髋臼标签。权重和训练影像不在仓库中。

1. 将有权使用的匹配权重放到 `PharmaLLM_V1.1/model/checkpoint_best.pth`；它被 Git 忽略。
2. 在独立环境安装与该权重相匹配的 nnU-Net / PyTorch。设置 `.env` 的 `NNUNET_PYTHON` 为该环境 Python 的绝对路径。
3. 激活 API `.venv`，在 `agent_app` 中准备部署元数据：

```bash
python scripts/prepare_segmentation_model.py
```

该命令检查元数据、生成布局与权重链接，不加载神经网络。其他权重会改变版本或标签；必须同步 `configs/segmentation.yaml`，不能仅重命名一个不同的 checkpoint。

私有 `.env` 中设置：

```dotenv
SEGMENTATION_BASE_URL=http://127.0.0.1:8092
NNUNET_MODEL_FOLDER=../model/femoralhead-t1
NNUNET_PYTHON=/path/to/nnunet-env/bin/python
SEGMENTATION_LOAD_MODE=resident
```

保留初始化生成的随机 `SEGMENTATION_API_KEY`，API、Worker 与私有分割服务使用一致配置。

**终端 C：** 激活 API `.venv`，进入 `agent_app`：

```bash
python scripts/start_worker.py --seg-gpu 1
```

GPU 1 同样只是示例，按分配选择。此命令先启动私有分割 HTTP 服务及独立 nnU-Net 常驻进程，真实加载成功后启动 Celery；后续病例复用模型。不要同时再启动另一套同端口分割服务。

只读检查：

```bash
python scripts/check_segmentation.py
```

常驻会保留模型权重，但推理工作区仍需额外显存。共享 GPU 需要调度或团队资源约定，启动脚本无法保证其他程序不占卡。详见[常驻与诊断](../PharmaLLM_V1.1/docs/agent/resident_segmentation.md)。

## 4. 验证

应用环境安装开发依赖后运行 CPU 检查：

```bash
python -m pip install --no-deps -e '.[dev]'
ruff check .
pytest -q -m 'not integration'
node --test tests/web_ui.test.cjs
```

完整集成测试需要名称以 `_test` 结尾的独立 PostgreSQL 数据库和独立 Redis 配置，见已有 CI。不要把 `TEST_DATABASE_URL` 指向业务库。

分割质量、真实 GPU 常驻峰值显存、吞吐与公网体验需要在部署环境单独验证；CPU 测试不能替代这些结果。
