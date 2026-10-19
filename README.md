# UAV Image Compression — Learned Model Training Code

Trains latent autoencoders with perceptual loss
(MSE + LPIPS + color loss) on 512x512x3 raw RGB images.

- `src/ae.py`: CNN autoencoder. Select the latent size with
  `LATENT_CONFIG` (16, 32, or 64).
- `src/mb.py`: MobileNetV2 autoencoder. Select width with
  `WIDTH_SELECT` (0.5 or 1.0) and latent size with `LATENT_CONFIG`
  (16, 32, or 64).
- `src/cheng_attn.py`: Cheng2020 Attention fine-tuning. Select quality
  with `QUALITY_SELECT` (3 or 6).
- `src/cheng_anchor.py`: Cheng2020 Anchor fine-tuning. Select quality
  with `QUALITY_SELECT` (3 or 6).

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place your `*.rgb` files (512x512x3, one byte per channel, row-major)
in the directory specified by `YOUR_DATASET_DIR` in the script you
intend to run.

## Run

```bash
python src/ae.py
python src/mb.py
python src/cheng_attn.py
python src/cheng_anchor.py
```

Trained encoder/decoder models, `metrics.csv`,
`training_config.json`, and model ZIPs are written to `./output/`
(or `YOUR_MODEL_EXPORT_DIR`).
