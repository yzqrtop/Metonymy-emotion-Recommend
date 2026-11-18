# 🧠 MetonymyRec

Metonymy-aware generative multimodal recommendation.

## 🖥️ Environment

- 🐍 Python 3.12
- 🔥 CPU profile: `torch`, `numpy`, `Pillow`, `PyYAML`, `pandas`
- 🚀 Paper profile: `transformers`, CUDA

### 📦 Install

```bash
python -m pip install -r requirements.txt
```

### ⚙️ Profiles

- 💻 `configs/cpu.yaml`: lightweight CPU backends
- 🎛️ `configs/paper.yaml`: Qwen2.5 + ViT + LoRA backends
- 🖼️ `configs/real_cpu.yaml`: official real data + Qwen2.5-0.5B + ViT; see `RUNBOOK.md`

## 📥 Datasets (download required)

Real experiments need the **official** datasets below. Nothing is auto-substituted: if a file is missing the build/test fails instead of falling back to synthetic data. Download first, then build with `scripts/build_real_data.py`.

| Dataset | Download | Status |
|---|---|---|
| 🖼️ **EmoMeta** | https://github.com/DUTIR-YSQ/EmoMeta (official repo: `DataAll.xlsx` + `dataF/*.jpg`) | Auto-downloadable |
| 🛒 **Taobao-MM** | https://huggingface.co/datasets/TaoBao-MM/Taobao-MM (official shards + SCL int8 feature map) | Auto-downloadable |
| 🍽️ **Yelp Open Dataset** | https://www.yelp.com/dataset (must accept the license manually) | **Manual download** |

```bash
# EmoMeta (official annotation Excel + bundled real images)
python scripts/download_real_data.py --dataset emometa

# Taobao-MM (one shard + item metadata + sorted SCL key index)
python scripts/download_real_data.py --dataset taobao --taobao-keys

# Yelp: accept the license on yelp.com, then place the archive / extracted files in data/raw/yelp/
python scripts/download_real_data.py --dataset yelp   # only checks local files, never bypasses the license
```

After downloading, build the processed real data:

```bash
# point --config at a profile whose paths.raw_dir contains the downloaded files above
python scripts/build_real_data.py --config <real-profile.yaml> --dataset all --taobao-items 64 --taobao-users 32
```

> ⚠️ Yelp is **not** auto-downloaded or replaced by seed data. Without the official archive, real Yelp training/evaluation cannot run; `download_real_data.py` only checks the local files.

## ▶️ Run

### 1️⃣ Prepare datasets

```bash
python scripts/prepare_data.py --config cpu
```

This builds the lightweight CPU seed data used by the `cpu` profile. For the official real datasets above, use the download/build commands in this section and `RUNBOOK.md`.

### 2️⃣ Run annotation

```bash
python scripts/annotate.py --config cpu --dataset all
```

### 3️⃣ Train a model

```bash
python scripts/train.py --config cpu --variant full --seed 0 --stage both
```

### 4️⃣ Evaluate a model

```bash
python scripts/evaluate.py --config cpu --variant full --seed 0 --tasks business,user,sentiment
```

### 5️⃣ Run all experiments

```bash
python scripts/run_experiments.py --config cpu
```

Reuse trained checkpoints:

```bash
python scripts/run_experiments.py --config cpu --skip-train
```

### 6️⃣ Smoke test

```bash
python scripts/smoke_test.py
```

## 📁 Outputs

- 📊 `outputs/results.json`
- 🧪 `outputs/ablation.json`
- 📑 `outputs/tables/tables.md`
- 📄 `outputs/tables/tables.tex`
- ✅ `outputs/annotation/`
- 🤝 `outputs/human_eval/`
