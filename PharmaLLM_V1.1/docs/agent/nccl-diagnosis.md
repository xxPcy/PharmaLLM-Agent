# 双卡初始化挂起排查（2026-10-21）

## 已验证的事实

- 当前单卡：API 主进程 PID 1950793、Engine PID 1954395，使用物理 GPU 6，端口 8002。只读检查 `/health` 返回 200，`/v1/models` 返回 200 且包含 `pharmallm-v1`；未提交生成请求或运行 GPU 通信测试。
- 原双卡两次启动留下 Worker `1861172/1861173` 和 `1938333/1938334`。它们均属于当前用户，PPID 为 1，主进程已退出，每个约占 704MiB。第一次启动残留在检查时已经运行约 37 分钟并持续消耗 CPU，属于未退出的孤儿进程，不是正常待机服务。
- 已核对 UID、PPID、进程组和启动时间，只向上述四个 Worker 发送 TERM；两秒后仍不响应，再发送 KILL。复查四个 PID 已消失，未触碰当前单卡进程或其他用户任务。
- 旧日志最后出现 `vLLM is using nccl==2.29.7`。本机 vLLM 0.30.0 的 `pynccl.py` 在此之后广播 NCCL ID、创建 communicator，然后执行一次 all-reduce 和 stream synchronization；这些步骤缺少逐步完成日志，因此现有输出不能再精确区分是哪一步阻塞。
- GPU 6/7 为同 NUMA 节点的 PCIe PIX 路径，拓扑未显示 NVLink。`nvidia-smi topo -p2p r` 对两张卡显示 OK，但这是能力查询，不是实际传输测试。
- `/sys/kernel/iommu_groups` 有分组，尚未核实 GPU 路径的 ACS/IOMMU 配置及其实际影响。未修改 BIOS、驱动、系统 PCIe 设置或重置 GPU。
- 加载的是当前 Conda 环境的 NCCL、CUDA 13 runtime 和系统 NVIDIA 580.95.05 驱动；单卡可用不能排除其中某个多卡通信路径的兼容问题。

## 目前能下的结论

问题范围已缩小到双卡通信初始化路径。单卡成功表明当前模型、基本 CUDA 加载和 HTTP 服务路径可用，暂不支持“模型权重损坏”或“单纯显存不足”的判断。P2P/IPC、NCCL 内存分配路径、网络选择和系统拓扑配置仍需对照日志验证，**尚未证明具体硬件或驱动根因**。

Ctrl+C 残留是另一项已确认的问题：原脚本用 exec 替换自身，没有独立监督器；vLLM 父进程退出后，卡在底层初始化中的 Worker 没有退出。启动器现在为每次新服务建立独立进程组，Ctrl+C/TERM/HUP 后先 TERM，最多等五秒，再 KILL 该组。它不会按进程名全局清理。修改从下一次启动生效；当前已运行单卡进程继续使用原有生命周期。

进程管理修复已通过 `pytest -q tests/test_process_control.py` 的 3 项 CPU 专项测试；未启动任何新的模型进程。清理旧 Worker 后再次检查，单卡服务的两个健康接口仍返回 200。

## 后续双卡对照（由用户执行）

先保持现有单卡服务用于开发。若需要验证双卡，**先停止当前单卡服务并确认 GPU 6/7 的剩余显存**，不能在同一张卡仍占用约 40GiB 时再启动预算 40GiB 的双卡实例。

第一步：仅禁用 NCCL P2P 传输并打开初始化日志，其他条件保持相同：

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET \
NCCL_P2P_DISABLE=1 \
VLLM_PORT=8002 \
GPU_MAX_USED_MIB=30000 \
GPU_MEMORY_UTILIZATION=0.50 \
python scripts/serve_vllm.py --gpus 6,7 --tp 2
```

若该组可启动而原始配置稳定挂起，则 P2P 路径是强相关因素，但仍不能仅凭此确定 ACS、IOMMU 或驱动中的哪项有问题。如果仍卡住，需要保留从 NCCL 初始化开始的 INFO 日志，不要不断重试累积进程。

后续可分别对照旧 V1.0 曾使用的 `NCCL_CUMEM_ENABLE=0` 等设置；每次只更改一项，旧脚本的设置不代表在当前 vLLM/NCCL 版本中已验证有效。若 NCCL 初始化通过后仍在 vLLM 自定义通信初始化中卡住，可添加 `--disable-custom-all-reduce`，该选项与 `NCCL_P2P_DISABLE` 的作用不同。

需要更深入验证时，单独安排 GPU 通信测试时间，再运行 nccl-tests/带宽测试及管理员级 ACS/IOMMU 检查。本次未运行这些测试。

参考：[NVIDIA NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)。
