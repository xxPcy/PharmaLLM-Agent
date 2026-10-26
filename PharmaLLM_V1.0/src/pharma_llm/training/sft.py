from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    __version__ as transformers_version,
    set_seed,
)

from pharma_llm.utils.config import get_required, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT for PharmaLLM, with optional LoRA/PEFT adapters.")
    parser.add_argument("--model-config", required=True, help="Path to configs/model.yaml")
    parser.add_argument("--train-config", required=True, help="Path to configs/train_sft.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run a short smoke test.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or omit to start normally.",
    )
    return parser.parse_args()


def resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def filter_supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def get_last_assistant_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("Record has no assistant message.")


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    def fallback_chatml() -> str:
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except ValueError:
            return fallback_chatml()
    except ValueError:
        return fallback_chatml()


def build_tokenizer(model_cfg: dict[str, Any]) -> Any:
    model_section = model_cfg.get("model", {})
    tokenizer_section = model_cfg.get("tokenizer", {})

    tokenizer = AutoTokenizer.from_pretrained(
        get_required(model_cfg, "model.name_or_path"),
        trust_remote_code=bool(model_section.get("trust_remote_code", False)),
        use_fast=bool(tokenizer_section.get("use_fast", True)),
    )
    tokenizer.padding_side = str(tokenizer_section.get("padding_side", "right"))
    tokenizer.truncation_side = str(tokenizer_section.get("truncation_side", "right"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer: Any, *, max_length: int) -> Dataset:
    def tokenize_one(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        assistant_index = get_last_assistant_index(messages)

        prompt_messages = messages[:assistant_index]
        full_text = apply_chat_template(tokenizer, messages[: assistant_index + 1], add_generation_prompt=False)
        prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        input_ids = full["input_ids"]
        labels = list(input_ids)
        prompt_length = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    return dataset.map(
        tokenize_one,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT data",
    )


def select_for_dry_run(dataset: Dataset, max_samples: int) -> Dataset:
    return dataset.select(range(min(max_samples, len(dataset))))


def load_sft_datasets(train_cfg: dict[str, Any], tokenizer: Any, *, dry_run: bool) -> tuple[Dataset, Dataset]:
    data_cfg = get_required(train_cfg, "data")
    training_cfg = get_required(train_cfg, "training")
    dry_cfg = train_cfg.get("dry_run", {})

    raw_train = load_dataset("json", data_files=str(data_cfg["train_file"]), split="train")
    raw_eval = load_dataset("json", data_files=str(data_cfg["validation_file"]), split="train")

    if dry_run:
        raw_train = select_for_dry_run(raw_train, int(dry_cfg.get("max_train_samples", 100)))
        raw_eval = select_for_dry_run(raw_eval, int(dry_cfg.get("max_eval_samples", 20)))

    max_length = int(training_cfg.get("max_seq_length", 4096))
    train_dataset = tokenize_dataset(raw_train, tokenizer, max_length=max_length)
    eval_dataset = tokenize_dataset(raw_eval, tokenizer, max_length=max_length)
    return train_dataset, eval_dataset


def build_model(model_cfg: dict[str, Any]) -> Any:
    model_section = get_required(model_cfg, "model")
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_section.get("trust_remote_code", False)),
    }
    dtype = resolve_dtype(model_section.get("dtype", model_section.get("torch_dtype")))
    if dtype is not None:
        dtype_key = "dtype" if Version(transformers_version) >= Version("4.56.0") else "torch_dtype"
        model_kwargs[dtype_key] = dtype
    attn_implementation = model_section.get("attn_implementation")
    if attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        warnings.warn(
            "flash_attention_2 is configured but flash_attn is not installed; "
            "falling back to sdpa. Install flash-attn for the intended training setup.",
            RuntimeWarning,
            stacklevel=2,
        )
        attn_implementation = "sdpa"
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_section["name_or_path"],
        **model_kwargs,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def maybe_apply_lora(model: Any, train_cfg: dict[str, Any]) -> Any:
    lora_cfg = train_cfg.get("lora", {})
    if not bool(lora_cfg.get("enabled", False)):
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install it with: pip install peft") from exc

    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(target_modules),
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(model, config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def build_training_args(train_cfg: dict[str, Any], *, dry_run: bool) -> TrainingArguments:
    training_cfg = get_required(train_cfg, "training")
    output_cfg = get_required(train_cfg, "output")
    hardware_cfg = train_cfg.get("hardware", {})
    dry_cfg = train_cfg.get("dry_run", {})

    output_dir = str(output_cfg["output_dir"])
    logging_dir = str(output_cfg["logging_dir"])
    summary_path = str(output_cfg.get("summary_path", "reports/training_summary.json"))
    if dry_run:
        output_dir = f"{output_dir}-dry-run"
        logging_dir = f"{logging_dir}-dry-run"
        summary_path = str(Path(summary_path).with_name("training_summary_dry_run.json"))

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training_cfg.get("learning_rate", 2.0e-5)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 1)),
        "bf16": bool(training_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_steps": int(training_cfg.get("save_steps", 500)),
        "eval_steps": int(training_cfg.get("eval_steps", 500)),
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_strategy": "steps",
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": "cosine",
        "report_to": training_cfg.get("report_to", ["tensorboard"]),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "save_total_limit": int(training_cfg.get("save_total_limit", 1)),
        "load_best_model_at_end": False,
        "overwrite_output_dir": bool(dry_run),
    }

    if dry_run:
        kwargs["max_steps"] = int(dry_cfg.get("max_steps", 10))
        kwargs["save_steps"] = int(dry_cfg.get("max_steps", 10))
        kwargs["eval_steps"] = max(1, min(5, int(dry_cfg.get("max_steps", 10))))
        kwargs["logging_steps"] = 1

    deepspeed_config = hardware_cfg.get("deepspeed_config")
    if deepspeed_config:
        kwargs["deepspeed"] = str(deepspeed_config)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    supported_kwargs = filter_supported_kwargs(TrainingArguments, kwargs)
    args = TrainingArguments(**supported_kwargs)
    setattr(args, "pharma_summary_path", summary_path)
    return args


def build_trainer(
    *,
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
    data_collator: Any,
) -> Trainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    trainer_parameters = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def write_summary(
    trainer: Trainer,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    *,
    train_count: int,
    eval_count: int,
    dry_run: bool,
) -> None:
    if not trainer.is_world_process_zero():
        return

    summary = {
        "dry_run": dry_run,
        "train_samples": train_count,
        "eval_samples": eval_count,
        "output_dir": trainer.args.output_dir,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    summary_path = Path(getattr(trainer.args, "pharma_summary_path"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml_config(args.model_config)
    train_cfg = load_yaml_config(args.train_config)

    set_seed(int(train_cfg.get("seed", 42)))

    tokenizer = build_tokenizer(model_cfg)
    train_dataset, eval_dataset = load_sft_datasets(train_cfg, tokenizer, dry_run=args.dry_run)
    training_args = build_training_args(train_cfg, dry_run=args.dry_run)
    model = maybe_apply_lora(build_model(model_cfg), train_cfg)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_metrics = train_result.metrics
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    write_summary(
        trainer,
        train_metrics,
        eval_metrics,
        train_count=len(train_dataset),
        eval_count=len(eval_dataset),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
