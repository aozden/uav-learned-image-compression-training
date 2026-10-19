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

## References

The Cheng2020 Anchor and Attention models are provided by
[CompressAI](https://github.com/InterDigitalInc/CompressAI).

If you use this code, please cite:

- Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7939–7948, 2020.

- J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, “CompressAI: a PyTorch Library and Evaluation Platform for End-to-End Compression Research,” *arXiv preprint arXiv:2011.03029*, 2020.

- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 586–595, 2018.
