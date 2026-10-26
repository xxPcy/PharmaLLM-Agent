#!/usr/bin/env python3
"""Dedicated nnU-Net environment: one case or a private, resident JSON-line worker."""

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import traceback

PREFIX = "PHARMA_RESIDENT:"


def emit(stage):
    print("PHARMA_STAGE:" + stage, flush=True)


def event(kind, **values):
    print(PREFIX + json.dumps({"event": kind, **values}), flush=True)


def failure_code(exc):
    message = str(exc).lower()
    return (
        "CUDA_OUT_OF_MEMORY"
        if ("cuda" in message and "out of memory" in message) or "cudaerrormemoryallocation" in message
        else "NNUNET_INFERENCE_FAILED"
    )


def bind_parent_lifetime():
    """Linux: prevent a resident CUDA child surviving an abruptly killed HTTP parent."""
    expected = os.environ.get("PHARMA_PARENT_PID")
    if expected and sys.platform == "linux":
        import ctypes
        import signal

        if os.getppid() != int(expected):
            raise SystemExit("Inference parent has exited")
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise RuntimeError("Unable to bind inference process lifetime")
        if os.getppid() != int(expected):
            raise SystemExit("Inference parent has exited")


class ModelSession:
    def __init__(self, folder, device, preload=False):
        # Never imported by the API or Celery Python.
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.export_prediction import export_prediction_from_logits

        self.torch, self.export = torch, export_prediction_from_logits
        self.device = device
        self.manifest = json.loads((folder / "model.json").read_text())
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if device == "cuda":
            print(
                "PHARMA_DEVICE:"
                + json.dumps(
                    {
                        "physical_gpu": os.environ.get("NNUNET_GPU_INDEX", "unknown"),
                        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "torch_device": "cuda:0",
                    }
                ),
                flush=True,
            )
            self.memory()
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=device == "cuda",
            device=torch.device(device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(folder), [self.manifest["fold"]], self.manifest["checkpoint"]
        )
        if preload:
            # Initialization itself restores CPU weights. Move them now, not on the first request.
            self.predictor.network = self.predictor.network.to(torch.device(device))
            self.predictor.network.eval()
            if device == "cuda":
                torch.cuda.synchronize()
                self.memory()

    def memory(self):
        if self.device != "cuda":
            return {}
        free, total = self.torch.cuda.mem_get_info()
        values = {
            "free_mib": free // 2**20,
            "total_mib": total // 2**20,
            "allocated_mib": self.torch.cuda.memory_allocated() // 2**20,
        }
        print("PHARMA_CUDA_MEMORY:" + json.dumps(values), flush=True)
        return values

    def predict(self, input_path, output, phase):
        phase("preprocessing")
        predictor = self.predictor
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        data, _, properties = preprocessor.run_case(
            [str(input_path)],
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        phase("inference")
        self.memory()
        with self.torch.inference_mode():
            logits = predictor.predict_logits_from_preprocessed_data(self.torch.from_numpy(data)).cpu()
        phase("postprocessing")
        self.export(
            logits,
            properties,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(output),
            False,
        )


def resident_loop(session, root, stream=None):
    stream = stream or sys.stdin
    event("ready", model_id=session.manifest["id"], model_version=session.manifest["version"])
    for line in stream:
        request_id = None
        try:
            if len(line) > 16384:
                raise ValueError("IPC message too large")
            item = json.loads(line)
            request_id = item["request_id"]
            if not isinstance(request_id, str) or not re.fullmatch(r"seg_[a-f0-9]{32}", request_id):
                raise ValueError("Invalid task ID")
            folder = (root / request_id).resolve()
            source, output = Path(item["input"]).resolve(), Path(item["output"]).resolve()
            if folder.parent != root or source != folder / "image_0000.nii.gz" or output != folder / "mask":
                raise ValueError("IPC path outside task folder")
            session.predict(source, output, lambda phase: event("phase", request_id=request_id, phase=phase))
            event("completed", request_id=request_id)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)  # Consumed only by the private resident log.
            event("failed", request_id=request_id, error_code=failure_code(exc))
        finally:
            gc.collect()
            if session.device == "cuda":
                session.torch.cuda.empty_cache()  # Releases workspace; the live network stays on GPU.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--task-root", type=Path)
    args = parser.parse_args()
    if args.resident and args.task_root is None:
        parser.error("--resident requires --task-root")
    if not args.resident and (args.input is None or args.output is None):
        parser.error("One-case mode requires --input and --output")
    if args.resident:
        bind_parent_lifetime()
    session = ModelSession(args.model_folder, args.device, preload=args.resident)
    if args.resident:
        resident_loop(session, args.task_root.resolve())
    else:
        session.predict(args.input, args.output, emit)
        emit("completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        code = failure_code(exc)
        if code == "CUDA_OUT_OF_MEMORY":
            print("PHARMA_ERROR:CUDA_OUT_OF_MEMORY", flush=True)
        if "--resident" in sys.argv:
            event("startup_failed", error_code=code if code == "CUDA_OUT_OF_MEMORY" else "MODEL_LOAD_FAILED")
        raise
