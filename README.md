```markdown
# Latent Diffusion Model Training on A100

A PyTorch Lightning–based implementation of latent diffusion training for text-to-image synthesis, optimized for NVIDIA A100 GPUs (CUDA 11.3). This repository builds on the CompVis *taming-transformers* codebase and integrates CLIP, nomi, and Hugging Face’s diffusers toolkit to train high-quality image generative models.

---

## 🚀 Features

- **Config-driven** training & evaluation via YAML + CLI overrides  
- **PyTorch Lightning** for scalable multi-GPU (**DDP**) training  
- Support for **text→image** datasets (iterable & map-style)  
- Automatic logging and checkpointing (TestTube, Weights & Biases)  
- Image & memory-usage callbacks for monitoring A100 performance  
- Example notebooks, scripts, and demo interfaces (Streamlit, Gradio)

---

## 📋 Requirements

- **Hardware**: NVIDIA GPU(s) with **CUDA 11.3** (A100 recommended)  
- **Python** 3.8+  
- **Dependencies** (see `requirements.txt` for exact versions):
  - `torch==1.12.1`, `torchvision==0.13.1`  
  - `pytorch-lightning==1.4.2`, `omegaconf==2.1.1`  
  - `transformers==4.22.2`, `diffusers==0.3.0`  
  - `taming-transformers`, `CLIP`, `nomi` (editable installs)  
  - Plus: `albumentations`, `opencv-python`, `einops`, `torchmetrics`, `gradio`, `streamlit`, etc.

---

## 🔧 Installation

1. **Clone** the repo  
   ```bash
   git clone https://github.com/GurucharanSavanth/GurucharanSavanth-Model-Traning-A100-V.git
   cd GurucharanSavanth-Model-Traning-A100-V
   ```
2. **Create & activate** your Python environment  
   ```bash
   conda create -n ldm-a100 python=3.9 -y
   conda activate ldm-a100
   ```
3. **Install** requirements  
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Verify** CUDA availability  
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

---

## 📂 Repository Structure

```text
├── assets/               # Static images & media
├── configs/              # YAML configs for model, data, trainer
├── data/                 # Dataset definitions & loaders
├── examples/             # Scripted usage examples
├── im-examples/          # Sample generated images
├── ldm/                  # Core latent diffusion modules
├── models/               # Pretrained & checkpointed models
├── notebooks/            # Jupyter tutorials & demos
├── scripts/              # Utility data/scripts (e.g., preprocessing)
├── main.py               # CLI entry point for train/eval
├── notebook_helpers.py   # Helpers for interactive notebooks
├── requirements.txt      # Exact package versions
└── setup.py              # Installable package definition
```

---

## ⚙️ Usage

### Training

```bash
python main.py \
  --base configs/latent-diffusion/txt2img-1p4B-eval.yaml \
  -t --project my_ldm_project \
  --logdir logs \
  --scale_lr True \
  --gpus 0,1,2,3
```

- `-t, --train` → start training  
- `--base` → list of YAML config files (merged in order)  
- `--gpus` → comma-separated GPU IDs (e.g. `0,1,2,3`)  
- Override any config key via `nested.key=value`, e.g.  
  ```bash
  python main.py --base configs/model.yaml train.params.batch_size=16
  ```

### Evaluation

```bash
python main.py \
  --resume logs/2025-04-24T12-00-00_mycfg/checkpoints/last.ckpt \
  --no-test
```

---

## 🔍 Configuration

All configurable parameters live in `configs/`. A typical config defines:

- **model** → architecture, base learning rate  
- **data** → dataset class & paths, batch size  
- **lightning** → Trainer arguments (accelerator, max_epochs…)  
- **callbacks** → custom logging, checkpoint callbacks  

Use `OmegaConf.to_yaml(config)` & your CLI to inspect & tweak.

---

## 📓 Notebooks & Demos

- **`notebooks/`**: walkthroughs for dataset loading, training loops, inference  
- **`examples/`**: quick-start scripts for sampling & model inspection  
- **`streamlit/`**, **`gradio/`** demos (see `requirements.txt`) showcase interactive model UIs.

---

## 🤝 Contributing

1. Fork & create a feature branch  
2. Write tests or validate example notebooks  
3. Open a pull request with clear description & related config changes  
4. Ensure all CI checks pass before merge

---


> Questions or feedback? Open an issue or reach out via GitHub Discussions.  
> Happy training on your A100 cluster! 🚀

