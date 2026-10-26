#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
from importlib.metadata import distribution
from pathlib import Path
import shutil
import subprocess

from env_utils import load
from process_control import run_supervised

parser = argparse.ArgumentParser(description="显式选择 A800 GPU，检查显存，然后启动独立 vLLM 服务")
parser.add_argument("--profile", choices=["a800-single.env", "a800-tp4.env", "a800-tp8.env"])
parser.add_argument("--gpus", help="物理 GPU 编号/UUID，逗号分隔；auto 表示自动选择空闲卡")
parser.add_argument(
    "--tp", type=int, choices=[1, 2, 4, 8], help="张量并行卡数；auto 时默认使用 Profile 的 TP"
)
parser.add_argument(
    "--disable-custom-all-reduce", action="store_true", help="多卡排障时禁用 vLLM 自定义 all-reduce"
)
args = parser.parse_args()
inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
load(args.profile)
model = Path(os.environ["MODEL_PATH"]).resolve()
if not (model / "config.json").is_file():
    raise SystemExit("MODEL_PATH 下没有 config.json，请指定完整的合并后 SFT 模型目录")
if not any(model.glob("model*.safetensors")) and not any(model.glob("pytorch_model*.bin")):
    raise SystemExit("MODEL_PATH 下未发现模型权重；不能使用仅有 LoRA adapter 的目录")
config = json.loads((model / "config.json").read_text())
auto_select = args.gpus == "auto"
devices = [
    item.strip()
    for item in (args.gpus if args.gpus and not auto_select else os.environ["CUDA_VISIBLE_DEVICES"]).split(
        ","
    )
    if item.strip()
]
tp = args.tp or (len(devices) if args.gpus and not auto_select else int(os.environ["TENSOR_PARALLEL_SIZE"]))
if tp not in {1, 2, 4, 8} or (not auto_select and (len(set(devices)) != len(devices) or len(devices) != tp)):
    raise SystemExit("CUDA_VISIBLE_DEVICES 必须明确且无重复，设备数量须等于 TP（1/2/4/8）")
for field in ["num_attention_heads", "num_key_value_heads"]:
    heads = config.get(field)
    if heads and heads % tp != 0 and tp % heads != 0:
        raise SystemExit(f"模型 {field}={heads} 与 TP={tp} 不兼容")
utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
if not 0.1 <= utilization <= 0.95:
    raise SystemExit("GPU_MEMORY_UTILIZATION 须在 0.1–0.95 之间")
query_command = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]
if not auto_select:
    query_command.extend(["-i", ",".join(devices)])
query = subprocess.run(
    query_command,
    check=True,
    capture_output=True,
    text=True,
    timeout=10,
)
rows = list(csv.reader(io.StringIO(query.stdout)))
threshold = int(os.environ.get("GPU_MAX_USED_MIB", "2048"))
if auto_select:
    # An inherited mask may be a scheduler allocation: never expand it automatically.
    allowed = (
        None if inherited_devices is None else [v.strip() for v in inherited_devices.split(",") if v.strip()]
    )
    eligible = []
    for row in rows:
        index, uuid, name, total, used = [value.strip() for value in row]
        if allowed is not None and not any(index == token or uuid.startswith(token) for token in allowed):
            continue
        if int(used) <= threshold and int(total) - int(used) >= int(total) * utilization + 512:
            eligible.append(row)
    # Keep multi-card groups homogeneous. Prefer less occupied cards; GPU 3 is last for TP>1.
    groups = {}
    for row in eligible:
        groups.setdefault(row[2].strip(), []).append(row)
    candidates = []
    for group in groups.values():
        group.sort(
            key=lambda row: (tp > 1 and row[0].strip() == "3", int(row[4].strip()), int(row[0].strip()))
        )
        if len(group) >= tp:
            candidates.append(group[:tp])
    if not candidates:
        raise SystemExit(
            f"可见范围内没有足够的同型号空闲 GPU（需要 {tp} 张，已用显存上限 {threshold} MiB）。"
            "请用 nvidia-smi 查看；脚本不会结束已有进程。"
        )
    rows = min(candidates, key=lambda group: sum(int(row[4].strip()) for row in group))
    devices = [row[0].strip() for row in rows]
    print("自动选择物理 GPU：" + ",".join(devices), flush=True)
if len(rows) != len(devices):
    raise SystemExit("nvidia-smi 返回的 GPU 数量与选择不一致")
for index, uuid, name, total, used in rows:
    if int(used.strip()) > threshold:
        raise SystemExit(
            f"GPU {index.strip()} 正使用 {used.strip()} MiB，超过允许的 {threshold} MiB。"
            "请选择空闲 GPU；脚本不会结束已有进程。"
        )
    if int(total.strip()) - int(used.strip()) < int(total.strip()) * utilization + 512:
        raise SystemExit(f"GPU {index.strip()} 空闲显存不足，请降低利用率或换卡")
    print(f"GPU {index.strip()}: {name.strip()}, {used.strip()}/{total.strip()} MiB", flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
os.environ["TENSOR_PARALLEL_SIZE"] = str(tp)
if any(row[0].strip() == "3" for row in rows) and tp > 1:
    print(
        "历史记录：物理 GPU 3 曾出现 NCCL P2P 挂起。若重现，请先用单卡/避开 GPU 3；"
        "也可在排障时设置 NCCL_P2P_DISABLE=1。",
        flush=True,
    )
binary = shutil.which("vllm")
if not binary:
    raise SystemExit("当前环境没有 vllm 命令，请切换到独立的 vLLM 环境")
command = [
    binary,
    "serve",
    str(model),
    "--served-model-name",
    os.environ["PHARMA_VLLM_MODEL_NAME"],
    "--host",
    os.environ.get("VLLM_HOST", "127.0.0.1"),
    "--port",
    os.environ.get("VLLM_PORT", "8000"),
    "--tensor-parallel-size",
    str(tp),
    "--dtype",
    "bfloat16",
    "--max-model-len",
    os.environ.get("MAX_MODEL_LEN", "8192"),
    "--gpu-memory-utilization",
    str(utilization),
    "--max-num-seqs",
    os.environ.get("MAX_NUM_SEQS", "32"),
    "--enable-prefix-caching",
]
if args.disable_custom_all_reduce:
    command.append("--disable-custom-all-reduce")
# Read installed CLI definitions without importing vLLM or initializing CUDA.
# Older releases use --disable-log-requests; 0.30.0 uses a BooleanOptionalAction.
vllm_package = distribution("vllm")
argument_sources = []
for relative in [
    "vllm/engine/arg_utils.py",
    "vllm/entrypoints/openai/cli_args.py",
    "vllm/entrypoints/launchers/cli_args.py",
]:
    source_path = vllm_package.locate_file(relative)
    if source_path.is_file():
        argument_sources.append(source_path.read_text(encoding="utf-8"))
argument_source = "\n".join(argument_sources)
if '"--enable-log-requests"' in argument_source or "'--enable-log-requests'" in argument_source:
    command.append("--no-enable-log-requests")
elif '"--disable-log-requests"' in argument_source or "'--disable-log-requests'" in argument_source:
    command.append("--disable-log-requests")
else:
    raise SystemExit("无法识别此 vLLM 版本的请求日志开关，请检查安装版本和 CLI 定义")
template = model / "chat_template.jinja"
if template.is_file():
    command.extend(["--chat-template", str(template)])
# No native auto-tool-choice: this ordinary SFT model is routed through validated JSON prompts.
# No API key flag: local loopback service; protect network exposure with a gateway/firewall.
raise SystemExit(run_supervised(command))
