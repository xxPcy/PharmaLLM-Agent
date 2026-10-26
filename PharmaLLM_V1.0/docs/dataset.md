# Dataset

Raw datasets are not committed to git.

Place source files under:

```text
data/raw/huatuogpt/
data/raw/tcm/
data/benchmarks/cmb_exam/
data/benchmarks/medqa_zh/
```

The first implemented schema validator expects canonical JSONL records with
`id`, `source`, `category`, and `messages`.

## Download Workflow

Create local folders:

```bash
mkdir -p data/raw/huatuogpt data/raw/tcm
mkdir -p data/benchmarks/cmb_exam data/benchmarks/medqa_zh
```

HuatuoGPT SFT data can be downloaded from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download \
  FreedomIntelligence/HuatuoGPT-sft-data-v1 \
  HuatuoGPT_sft_data_v1.jsonl \
  --repo-type dataset \
  --local-dir data/raw/huatuogpt
```

CMB can be downloaded from the official GitHub repository:

```bash
git clone https://github.com/FreedomIntelligence/CMB.git /tmp/CMB
unzip /tmp/CMB/data/CMB.zip -d data/benchmarks/cmb_exam
```

MedQA-ZH is provided by the official MedQA repository via a Google Drive folder.
Download the Mainland China QA files and place the 4-option split files under:

```text
data/benchmarks/medqa_zh/
```

The exact TCM training source is intentionally not pinned yet. Once the source
dataset is selected, Phase 2 will add a source-specific normalizer that converts
it into the canonical JSONL schema.
