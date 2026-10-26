"""Inspect PyTorch checkpoint metadata without importing torch or rebuilding tensors."""

import collections
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import zipfile


class _Opaque:
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _encode(value, encoding):
    if encoding not in {"latin1", "ascii", "utf-8"}:
        raise ValueError("Unsupported checkpoint encoding")
    return value.encode(encoding)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("_codecs", "encode"):
            return _encode
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch._utils", "_rebuild_tensor"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy", "dtype"),
            ("torch", "device"),
        }
        allowed.update(
            ("torch", kind + "Storage")
            for kind in ("Float", "Long", "Double", "Int", "Byte", "Bool", "Half", "BFloat16")
        )
        if (module, name) in allowed:
            return _Opaque
        raise ValueError("Checkpoint contains unsupported metadata globals")

    def persistent_load(self, pid):
        # Never dereference tensor storage, device names, paths or executable globals.
        return _Opaque(pid)


def read_metadata(checkpoint):
    with zipfile.ZipFile(checkpoint) as archive:
        item = next(i for i in archive.infolist() if i.filename.endswith("/data.pkl"))
        if item.file_size > 2 * 1024 * 1024:
            raise ValueError("Checkpoint metadata exceeds limit")
        root = MetadataReader(io.BytesIO(archive.read(item))).load()
    args = root["init_args"]
    result = {
        "plans": args["plans"],
        "dataset_json": args["dataset_json"],
        "configuration": args["configuration"],
        "fold": args["fold"],
        "trainer": root["trainer_name"],
    }
    json.dumps(result, allow_nan=False)  # Only primitive, JSON-safe deployment metadata is accepted.
    return result


def prepare_model(checkpoint: Path, output: Path, modality="T1"):
    checkpoint = checkpoint.resolve(strict=True)
    data = read_metadata(checkpoint)
    if data["configuration"] != "3d_fullres" or len(data["dataset_json"]["channel_names"]) != 1:
        raise ValueError("This adapter requires a single-channel 3d_fullres model")
    checksum = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    fold = output / ("fold_" + str(data["fold"]))
    fold.mkdir(parents=True, exist_ok=True)
    destination = fold / "checkpoint_best.pth"
    if destination.exists() and destination.resolve() != checkpoint:
        raise ValueError("Refusing to replace an existing checkpoint")
    if not destination.exists():
        destination.symlink_to(os.path.relpath(checkpoint, fold.resolve()))
    for name, value in (("plans.json", data["plans"]), ("dataset.json", data["dataset_json"])):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    labels = {int(value): name for name, value in data["dataset_json"]["labels"].items()}
    manifest = {
        "id": "femoralhead-t1-best",
        "name": "股骨头与髋臼 · T1",
        "version": "sha256-" + checksum.hexdigest()[:20],
        "task_types": ["femoral_head_segmentation"],
        "modalities": [modality],
        "channels": 1,
        "labels": labels,
        "fold": data["fold"],
        "trainer": data["trainer"],
        "configuration": data["configuration"],
        "checkpoint": "checkpoint_best.pth",
        "sha256": checksum.hexdigest(),
    }
    (output / "model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
