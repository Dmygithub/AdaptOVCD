# AdaptOVCD Installation Guide

Training-free open-vocabulary change detection framework using foundation model synergy.

## Prerequisites

**System Requirements:**
- CUDA 12.1+
- Python 3.11+
- ~10GB disk space for model weights

### 1. Create Environment

```bash
# Create conda environment
conda create -n adaptovcd python=3.11 -c conda-forge -y
conda activate adaptovcd

# Navigate to project directory
cd /path/to/AdaptOVCD

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install opencv-python scikit-image pycocotools timm transformers==4.36.0 einops ftfy
```

---

## Three-Stage Model Installation

### Stage 1: Segmentation - **SAM-HQ** ⭐

High-quality instance segmentation with boundary refinement.

```bash
mkdir -p third_party
cd third_party

# 1. Install Original SAM (Required Base)
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..

# 2. Install SAM-HQ (Extension)
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq

# Copy HQ extension and pre-configured __init__.py (Critical Step!)
cp seginw/segment_anything/build_sam_hq.py segment_anything/
cp ../../patches/sam_hq_init.py segment_anything/__init__.py

# Install extension
pip install -e .
cd ../..

# Download SAM-HQ weights (2.39GB)
mkdir -p models/hqsam
mkdir -p models/sam

# Option A: SAM-HQ (Recommended)
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth \
     -O models/hqsam/sam_hq_vit_h.pth

# Or use mirror (if above fails):
wget https://hf-mirror.com/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth \
     -O models/hqsam/sam_hq_vit_h.pth

# Option B: Original SAM (Optional backup)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O models/sam/sam_vit_h_4b8939.pth
```

---

### Stage 2: Comparison - **DINOv3** ⭐

Self-supervised visual features for robust semantic change detection.

```bash
# Clone DINOv3 repository
cd third_party
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3
pip install -e .
cd ../..
```

**⚠️ Model Weights (Requires Application):**

DINOv3 weights require Meta approval. Follow these steps:

1. Visit: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. Fill out the application form
3. Wait for approval email (usually within 24 hours)
4. Download `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (~1.2GB) from the link in email
5. Place the file:
   ```bash
   mkdir -p models/dinov3
   mv dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth models/dinov3/
   ```

**Important:** Keep the original filename with hash suffix. Do not rename.

<details>
<summary><b>Alternative: DINOv2 (No Application Required)</b></summary>

Auto-downloads from HuggingFace on first run (~1.1GB), cached in `~/.cache/huggingface/hub/`

```yaml
comparator:
  variant: 'dinov2_vitl14'
  weights_path: null
```

</details>

<details>
<summary><b>Alternative: DINO (Lightweight)</b></summary>

Auto-downloads from torch.hub on first run, cached in `~/.cache/torch/hub/checkpoints/`

```yaml
comparator:
  variant: 'dino_vitb16'
  weights_path: null
```

</details>

**First Run:**
The model will be automatically downloaded (~1.1GB) when you run:
```bash
python demo.py --model OVCD_levircd
```

Weights will be cached in: `~/.cache/huggingface/hub/`

**Performance Note:** DINOv2 achieves comparable results to DINOv3 in most scenarios.

</details>

<details>
<summary><b>Alternative: DINO (Original, Lightweight)</b></summary>

Original DINO is the lightest option but with slightly lower performance.

```yaml
comparator:
  variant: 'dino_vitb16'
  weights_path: null  # Auto-download
```

</details>

---

### Stage 3: Identification - **DGTRS-CLIP** ⭐

Remote sensing CLIP for open-vocabulary semantic understanding.

```bash
# Clone OpenAI CLIP (required dependency)
cd third_party
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ..

# Clone DGTRS repository (required for longclip module)
git clone https://github.com/MitsuiChen14/DGTRS.git
cd ..

# Download DGTRS-CLIP weights (1.7GB)
mkdir -p models/DGTRS
wget https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-L-14/resolve/09dc0a7b6532f769aa64dae957bfbad068d2735f/LRSCLIP_ViT-L-14.pt \
     -O models/DGTRS/LRSCLIP_ViT-L-14.pt

# Or use mirror (if above fails):
wget https://hf-mirror.com/MitsuiChen14/DGTRS-CLIP-ViT-L-14/resolve/09dc0a7b6532f769aa64dae957bfbad068d2735f/LRSCLIP_ViT-L-14.pt \
     -O models/DGTRS/LRSCLIP_ViT-L-14.pt
```

**Note:** DGTRS-CLIP is a remote sensing fine-tuned version of CLIP from the [RS5M project](https://github.com/om-ai-lab/RS5M).
> **Version Note:** The link above points to the specific `LRSCLIP_ViT-L-14.pt` (March 2025 version) used in the paper, ensuring reproducibility. Manual download is available [here](https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-L-14/blob/09dc0a7b6532f769aa64dae957bfbad068d2735f/LRSCLIP_ViT-L-14.pt).

---

## Verification

After installation, verify your setup:

```bash
# Check directory structure
ls models/hqsam/sam_hq_vit_h.pth          # Should exist (2.39GB)
ls models/dinov3/*.pth                     # Should exist (1.2GB) or use DINOv2
ls models/DGTRS/LRSCLIP_ViT-L-14.pt       # Should exist (1.7GB)

# Quick test
python demo.py --help
```

**Expected Directory Structure:**

```
AdaptOVCD/
├── patches/                   # Pre-configured files
│   └── sam_hq_init.py
├── models/                    # Model weights (~5.3GB total)
│   ├── hqsam/
│   │   └── sam_hq_vit_h.pth
│   ├── dinov3/
│   │   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
│   └── DGTRS/
│       └── LRSCLIP_ViT-L-14.pt
├── third_party/               # Cloned repositories
│   ├── segment-anything/
│   ├── sam-hq/
│   ├── dinov3/
│   ├── CLIP/
│   └── DGTRS/
└── configs/
    └── models/
        └── OVCD_*.yaml        # Pre-configured models
```

## Quick Start

```bash
# Run demo on sample images
python demo.py \
  --model OVCD_levircd \
  --input1 demo_images/A/test_5.png \
  --input2 demo_images/B/test_5.png

# Run evaluation on LEVIR-CD dataset
python evaluate.py \
  --dataset levircd \
  --model OVCD_levircd \
  --save_predictions
```

