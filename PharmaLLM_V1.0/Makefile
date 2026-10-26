.PHONY: install prepare-data train eval-base eval-sft serve benchmark test

install:
	pip install -e ".[dev]"

prepare-data:
	bash scripts/prepare_data.sh

train:
	bash scripts/train_sft.sh

eval-base:
	bash scripts/eval_base.sh

eval-sft:
	bash scripts/eval_sft.sh

serve:
	bash scripts/serve_vllm.sh

benchmark:
	bash scripts/benchmark_serving.sh

test:
	pytest -q

