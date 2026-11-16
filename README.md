# Satellite Land Cover Segmentation & Change Detection

A U-Net (TensorFlow/Keras) pipeline that classifies satellite imagery into
land-cover classes and compares two dates to detect urban growth,
deforestation, flooding and bare-land expansion.

**Classes:** background · urban · vegetation · water · bare_land
(edit in `config.py`)

---

## Project structure

```
satellite_land_cover_segmentation/
├── config.py               # All paths, hyperparameters, classes, colors
├── dataset.py              # Loading, preprocessing, augmentation, tf.data
├── model.py                # U-Net architecture
├── losses.py               # Dice + cross-entropy loss
├── metrics.py              # IoU, Dice, precision, recall, F1, confusion matrix
├── train.py                # Training script
├── evaluate.py             # Test-set evaluation
├── predict.py              # Inference on new images
├── change_detection.py     # Multi-temporal change detection
├── visualization.py        # All plots and figures
├── verify_setup.py         # Pre-flight environment check
├── generate_sample_data.py # Synthetic demo data (smoke test only)
├── utils/helpers.py        # Shared helper functions
├── requirements.txt
└── data/                   # Your datasets (see below)
```

---

## 1. Setup

Requires **Python 3.9 – 3.11** (TensorFlow 2.13–2.16 does not support newer versions).

```bash
cd satellite_land_cover_segmentation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify the installation
python verify_setup.py
```

**Important:** always run commands from inside the project folder.

## 2. Prepare the data

**Option A – your own data (recommended).** Put matching files in:

```
data/raw/images/   tile_001.png, tile_002.png, ...
data/raw/masks/    tile_001.png, tile_002.png, ...
```

- Image and mask must share the same filename (extension may differ).
- Masks are single-channel; each pixel value is the class id `0–4`.
- Supported formats: PNG, JPG, TIF/TIFF.

`train.py` automatically splits `data/raw` into train / val / test
(70 / 15 / 15). Alternatively, fill `data/train|val|test/{images,masks}` yourself.

**Option B – quick demo with synthetic data** (verifies the pipeline only):

```bash
python generate_sample_data.py --samples 60
```

## 3. Train

```bash
python train.py
python train.py --epochs 40 --batch-size 16 --lr 0.0001
python train.py --resume          # continue from the best saved model
```

Outputs: `models/best_unet_model.h5`, `models/final_unet_model.h5`,
`outputs/training_history.csv`, `results/images/training_curves.png`.

## 4. Evaluate

```bash
python evaluate.py
```

Outputs: `outputs/evaluation_report.json`, `results/images/confusion_matrix.png`.

## 5. Predict on new images

```bash
python predict.py --input data/test/images --output outputs/predictions
python predict.py --input path/to/image.png
```

## 6. Change detection

```bash
python change_detection.py --before path/to/earlier.png --after path/to/later.png
```

Outputs in `outputs/change_detection/`: statistics JSON, change maps,
before/after segmentations and area charts. Set `PIXEL_RESOLUTION_M` in
`config.py` to your imagery's resolution for correct hectare values.

---

## Google Colab

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/satellite_land_cover_segmentation
!pip install -r requirements.txt
!python verify_setup.py
!python train.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: config` / `utils` | Run from inside the project folder |
| `No image/mask pairs found` | Add data to `data/raw/` (Step 2) |
| `Model file not found` | Run `python train.py` first |
| TensorFlow install fails | Use Python 3.9–3.11 |
| Out of memory | Lower `--batch-size` (e.g. 4) |
