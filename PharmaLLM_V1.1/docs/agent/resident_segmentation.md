# 分割服务诊断、终端 C 选卡与常驻模型 · 2026-10-26

## 本次诊断

本次只读检查发现：

- 应用 API `127.0.0.1:8082/health/ready` 返回 200。
- 语言模型 `127.0.0.1:8002/health` 返回 200。
- 配置的分割地址为 `http://127.0.0.1:8092`，连接失败，监听列表也没有 8092。
- 分割密钥已配置，nnU-Net Python 与模型元数据文件存在。没有输出密钥或读取病例像素。

因此，排查时的分割失败原因是独立分割服务没有运行。原 `python scripts/start_worker.py` 只启动 CPU Celery Worker，不会启动模型服务。旧的“分割服务不可用或返回格式不符合接口契约”把连接与协议错误混在一起，不能据此判断权重或接口格式出了问题。

截图中历史的“无法连接模型服务”属于语言模型链路；本次探测时该链路正常，不能用当前探测反推它在历史时刻的状态。

## 更新后的启动方式

影像工作区已移回对话右侧，保留原主题、四视图、关闭、拖动和结果自动更新。无需迁移数据库、安装新依赖或修改权重。

等待现有回答与任务结束，在原终端 B、C 停止旧进程，再启动。

终端 B：

```bash
cd /path/to/PharmaLLM/PharmaLLM_V1.1/agent_app
source .venv/bin/activate
APP_PORT=8082 python scripts/start_api.py
```

终端 C，一并启动常驻分割服务与 CPU Worker：

```bash
cd /path/to/PharmaLLM/PharmaLLM_V1.1/agent_app
source .venv/bin/activate

# 0 是示例，改成分配给分割的物理 GPU 编号。
python scripts/start_worker.py --seg-gpu 0
```

终端 C 会先打印物理 GPU、UUID 和空闲显存，通过独立 `onfh` Python 启动常驻模型；健康检查验证模型身份与版本，就绪后才启动 CPU Worker。应用和 Celery 不导入 Torch 或加载分割网络。看到“分割常驻模型已加载”和“分割服务已就绪”后再提交/重试分割。

浏览器 Ctrl+F5，原失败卡片可点“重试分割”，不需要重传仍有效的影像。旧失败记录不会被直接改写成成功；重试产生新的执行尝试。

终端 A 的 vLLM 无需为了本次变更重新加载。不要同时在另一终端再开相同 8092 分割服务；合并启动检测到端口被占用时会拒绝启动，不会结束已有进程。

## 可选参数

| 命令或参数 | 作用 |
| --- | --- |
| `python scripts/start_worker.py --seg-gpu 4` | 分割使用物理 GPU 4，默认常驻；CPU Worker 仍不占 GPU |
| `--seg-min-free-mib 8192` | 启动时至少有 8192 MiB 空闲显存，否则拒绝；不是预留，也不是实际峰值需求 |
| `--startup-timeout 600` | 给模型加载最多 600 秒；默认 300 秒 |
| `--seg-load-mode per-task` | 回到每个病例启动推理进程、结束后卸载的方式 |
| `--seg-cpu` | CPU 分割常驻，仅适合慢速排障 |
| `--seg-port 8092` | 显式端口，必须与 `SEGMENTATION_BASE_URL` 一致 |
| 不带 `--seg-gpu/--seg-cpu` | 保留旧行为：只启动 CPU Worker，分割服务由操作者另外维护 |

如果希望继续分开管理，先在独立终端启动并等待就绪：

```bash
python scripts/start_segmentation.py --gpu 0
```

然后在终端 C 执行不带选卡参数的 `python scripts/start_worker.py`。这是另一种部署方式，不与上面的合并方式同时使用。

合并启动要求分割根地址为本机 HTTP 回环地址，例如 `http://127.0.0.1:8092`，不能写成 `/v1`。远程服务仍可通过旧的独立 CPU Worker 模式连接。

## 常驻如何实现

```text
终端 C 的管理进程
  ├─ CPU Celery Worker
  └─ 私有分割 HTTP 服务
       └─ onfh Python 常驻进程
            启动：读取模型 → 加载权重 → network.to(指定 GPU) → 校验就绪
            任务：预处理 → 滑窗预测 → 原始空间导出 → 等待下一任务
```

新模式在启动时真正把网络移动到所选 GPU，并等待 CUDA 同步完成，才发出 ready 帧。后续任务复用同一模型对象和进程，不会每个病例重新初始化网络或重读权重文件。GPU UUID 保证进程内 `cuda:0` 对应选定的物理卡。

一次只执行一个分割，CPU Worker 的两个并发不会加载两套 GPU 模型。任务与源文件仍经过原有授权、幂等、mask 几何、单位和标签校验。

启动失败、进程崩溃、错误请求编号或错误协议不会伪装为 ready 或 completed。OOM 会失败为明确错误；可恢复的单次 OOM 不主动丢弃仍有效的模型对象。任务超时会关闭失去同步的常驻进程，服务显示未就绪，需要处理原因后重启。不会自动无限重启并抢占 GPU。

管理进程只记录并清理本次启动的子进程组。常驻 runner 在 Linux 下绑定 HTTP 父进程生命周期，避免父进程被强制结束后留下独立 CUDA 常驻进程。关闭终端 C 会释放本次分割模型；它不接管原先单独运行的服务。

## 常驻不等于显卡独占

常驻保留的是网络权重与仍有引用的运行状态，推理激活、滑窗输出和算子工作区还需要额外空间。nnU-Net 会释放可回收缓存；`empty_cache()` 也会让未占用缓存可供其他 GPU 应用使用，因此不能把空闲缓存当作固定预留。[PyTorch 2.10 文档](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html)

建议给分割分配一张明确的 GPU，并让其他训练/推理任务避开该卡。若要强制阻止其他 CUDA 程序进入，需要管理员或调度系统配置独占/资源隔离；NVIDIA 默认计算模式允许多个上下文，Exclusive Process 模式限制每卡一个上下文。脚本不会自行更改服务器的计算模式，也不会以占满整卡的方式冒充资源预留。[NVIDIA 计算模式说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html#compute-mode)

目前没有对这个 checkpoint 的常驻实际峰值显存作新测量，不能承诺固定预留多少 GiB 或保证共享卡永不 OOM。vLLM 的 `GPU_MEMORY_UTILIZATION` 不控制 nnU-Net。

## 如何检查与排障

只读检查，不提交病例或加载模型：

```bash
python scripts/check_segmentation.py
```

| 类别 | 含义与操作 |
| --- | --- |
| `SEGMENTATION_CONNECTION_FAILED` | 服务未运行、地址错误或连接中断；先启动终端 C 的分割服务 |
| `SEGMENTATION_LOADING` | 模型正在加载；等待就绪 |
| `SEGMENTATION_AUTH_FAILED` | API/Worker/服务密钥不一致；核对配置并重启相关进程 |
| `SEGMENTATION_ENDPOINT_NOT_FOUND` | 接口或任务不存在；检查根地址与服务是否正确 |
| `SEGMENTATION_PROTOCOL_ERROR` | 返回格式或协议版本不匹配；可能连到了其他 HTTP 服务 |
| `MODEL_VERSION_MISMATCH` | 分割服务的模型身份/版本与注册配置不匹配 |
| `SEGMENTATION_MODEL_NOT_READY` | 常驻模型加载失败或进程异常；查看终端 C 和私有日志 |
| `SEGMENTATION_CUDA_OUT_OF_MEMORY` | 显存不足；调整 GPU 分配后重启并重试 |
| `SEGMENTATION_BUSY` | 服务队列已满，等待已有任务 |
| `SEGMENTATION_CONNECTION_TIMEOUT` | HTTP 响应超时；检查分割服务状态 |

常驻日志：`agent_app/var/segmentation-service/resident.log`。每任务状态仍在对应 `seg_.../task.json`，单次进程模式继续保留 `runner.log`。日志只供本机操作者查看，不通过网页回传原始堆栈、路径或密钥。

## 验证范围

本次未启动真实 GPU 常驻模型或推理，未改写业务数据库、配置密钥或训练环境。

使用隔离 CPU 子进程和合成 NIfTI，检查一次加载后连续两任务保持同一 PID、OOM 后仍可接下一任务、加载失败、进程崩溃、协议串任务、几何/单位、健康状态与退出清理。以替身网络验证启动时调用 `.to(device)`，两任务不重复初始化；这些不替代真实 onfh/CUDA 运行验收。

另检查终端 C 选卡参数、端口冲突拒绝、等待模型就绪才启动 Worker、只清理自有子进程，以及连接/鉴权/格式错误的安全分类。浏览器使用真实前端与合成 API 检查右侧布局和状态提示，截图不是病例结果。

截图与机器可读结果见 [本轮验证](../../../docs/screenshots/README.md)。

本轮最终结果：**120 项 CPU、12 项 Node、98 项 Firefox 检查通过**；18 项 PostgreSQL/Redis 集成未运行。
