<div align="center">

# AdaptOVCD

### Training-Free Open-Vocabulary Change Detection via Adaptive Foundation Model Synergy

[Paper]() | [arXiv]()

</div>

## Overview

**AdaptOVCD** is a training-free framework for open-vocabulary change detection in remote sensing imagery. By synergistically integrating three foundation models with adaptive enhancement modules, AdaptOVCD enables zero-shot detection of arbitrary change categories specified via natural language—without any task-specific training or annotated data.

### Framework Overview

The pipeline implements dual-dimensional multi-level information fusion: vertically constructing a data-feature-decision cascade, and horizontally conducting targeted adaptive designs at each level.

![Figure 2: Framework Overview](fig/fig2.jpg)

### Visualization of Intermediate Processes

To gain a deeper understanding of the mechanism of each module, we visualize the intermediate processes of AdaptOVCD. The visualization demonstrates the progressive filtering effect, where spurious detections are correctly eliminated while authentic changes are retained.

![Figure 5: Intermediate Process Visualization](fig/fig5.jpg)

## Installation

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

## Project Structure

```
AdaptOVCD/
├── models/                                # See INSTALL.md for setup
├── third_party/                           # See INSTALL.md for setup
├── data/                                  # Download required
│   ├── levircd/                           # Ready to use
│   ├── Building change detection dataset_add/  # Preprocessing required
│   ├── DSIFN/                             # Preprocessing required
│   └── second_dataset/                    # Preprocessing required
├── configs/
│   ├── models/                            # Model configurations
│   └── datasets/                          # Dataset configurations
├── utils/
│   └── datasets-test/                     # Preprocessing scripts
│       ├── dsifn.py
│       ├── second.py
│       └── whucd.py
├── evaluate.py                            # LEVIR-CD, WHU-CD, DSIFN evaluation
├── evaluate_second.py                     # SECOND evaluation
└── demo.py                                # Demo script
```

## Datasets

We use **test sets only** for zero-shot evaluation. Download and organize as follows:

| Dataset | Type | Test Size | Resolution | Download |
|---------|------|-----------|------------|----------|
| LEVIR-CD | Building | 128 pairs | 0.5m | [Link](https://justchenhao.github.io/LEVIR/) |
| WHU-CD | Building | 690 pairs | 0.075m | [Link](http://gpcv.whu.edu.cn/data/building_dataset.html) |
| DSIFN | Building | 48 pairs | 2m | [Link](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) |
| SECOND | Semantic (6 classes) | 1000+ pairs | - | [Link](https://captain-whu.github.io/SCD/) |

### Raw Dataset Structure

After downloading, place datasets in `data/` directory:

```
data/
├── levircd/                          # LEVIR-CD (ready to use)
│   ├── A/
│   ├── B/
│   └── label/
├── Building change detection dataset_add/   # WHU-CD (needs preprocessing)
│   └── 1. The two-period image data/
│       ├── 2012/splited_images/test/image/
│       ├── 2016/splited_images/test/image/
│       └── change_label/test/
├── DSIFN/                            # DSIFN (needs preprocessing)
│   └── test/
│       ├── t1/
│       ├── t2/
│       └── mask/
└── second_dataset/                   # SECOND (needs preprocessing)
    └── test/
        ├── im1/
        ├── im2/
        ├── label1/
        └── label2/
```

### Dataset Preprocessing

Run preprocessing scripts to convert datasets into unified format (A/B/label):

```bash
# WHU-CD: Convert TIF to PNG and reorganize structure
python utils/datasets-test/whucd.py --source data/Building\ change\ detection\ dataset_add --output data/whucd

# DSIFN: Convert mask TIF to JPG and reorganize
python utils/datasets-test/dsifn.py --source data/DSIFN/test --target data/dsifn

# SECOND: Generate class-specific change labels from semantic labels
python utils/datasets-test/second.py
# Input: data/second_dataset/test -> Output: data/second/
```

### Preprocessed Dataset Structure

After preprocessing:

```
data/
├── levircd/          # Ready to use
│   ├── A/
│   ├── B/
│   └── label/
├── whucd/            # After whucd.py
│   ├── A/
│   ├── B/
│   └── label/
├── dsifn/            # After dsifn.py
│   ├── A/
│   ├── B/
│   └── label/
└── second/           # After second.py
    ├── im1/
    ├── im2/
    ├── label1/
    ├── label2/
    ├── label_building/
    ├── label_water/
    ├── label_tree/
    ├── label_low_vegetation/
    ├── label_non_veg_ground_surface/
    └── label_playground/
```

## Usage

### Demo

```bash
python demo.py \
    --model OVCD_levircd \
    --input1 demo_images/A/00004.png \
    --input2 demo_images/B/00004.png \
    --output outputs/demo/
```

### Evaluation

**Binary change detection** (LEVIR-CD, WHU-CD, DSIFN):

```bash
# LEVIR-CD
python evaluate.py --model OVCD_levircd --dataset levircd

# WHU-CD
python evaluate.py --model OVCD_whucd --dataset whucd

# DSIFN
python evaluate.py --model OVCD_dsifn --dataset dsifn
```

**Semantic change detection** (SECOND - 6 classes):

```bash
# Single class
python evaluate_second.py --class building --output_dir outputs/second/building

# All 6 classes
python evaluate_second.py --class all --output_dir outputs/second/all
```

Available SECOND classes: `building`, `water`, `tree`, `low_vegetation`, `non_veg_ground_surface`, `playground`

### Qualitative Results

**Building Change Detection** on LEVIR-CD, WHU-CD, DSIFN, and SECOND (Building):

![Figure 3: Building Change Detection Results](fig/fig3.jpg)

**Open-Vocabulary Change Detection** on SECOND across six semantic categories (Building, Low Vegetation, Non-veg. Ground, Playground, Tree, Water):

![Figure 4: Open-Vocabulary Results](fig/fig4.jpg)

### Save Predictions

Add `--save_predictions` flag to keep output masks:

```bash
python evaluate.py --model OVCD_levircd --dataset levircd --save_predictions --output_dir outputs/
```

## Citation

```bibtex
@article{adaptovcd2025,
  title={AdaptOVCD: Training-Free Open-Vocabulary Change Detection via Adaptive Foundation Model Synergy},
  author={},
  journal={},
  year={2025}
}
```

## Acknowledgements

- [SAM-HQ](https://github.com/SysCV/sam-hq) (Apache 2.0)
- [DINOv3](https://github.com/facebookresearch/dinov3) (Meta License)
- [DGTRS-CLIP](https://github.com/MitsuiChen14/DGTRS) (Apache 2.0)
- [CLIP](https://github.com/openai/CLIP) (MIT)
- [DynamicEarth](https://github.com/likyoo/DynamicEarth)

## License

Apache 2.0
