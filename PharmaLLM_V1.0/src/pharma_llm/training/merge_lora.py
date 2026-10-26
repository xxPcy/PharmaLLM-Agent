from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into its base model.")
    parser.add_argument("--adapter-path", required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument("--output-path", required=True, help="Path for the merged Hugging Face model.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Dtype used while loading and saving the merged model.",
    )
    parser.add_argument("--device-map", default="auto", help='Transformers device_map, e.g. "auto" or "cpu".')
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer

    args = parse_args()
    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, **model_kwargs)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)

    peft_config = PeftConfig.from_pretrained(adapter_path)
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)
    tokenizer.save_pretrained(output_path)

    print(f"Merged LoRA adapter from {adapter_path} into {output_path}")


if __name__ == "__main__":
    main()
