"""MobileNetV2 latent autoencoder with perceptual loss training."""

from pathlib import Path
import csv
import gc
import json
import math
import random
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import lpips


# ============================================================
# PARAMETERS - SET THESE BEFORE RUNNING
# ============================================================

# Local dataset folder holding 512x512x3 raw RGB files (*.rgb).
YOUR_DATASET_DIR = Path("./dataset_raw")

# Model export folder (checkpoints, TorchScript models, metrics, ZIPs).
YOUR_MODEL_EXPORT_DIR = Path("./output")

DATASET_DIR = YOUR_DATASET_DIR
OUTPUT_DIR = YOUR_MODEL_EXPORT_DIR

MAX_IMAGES = 2000

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANNELS = 3

BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 500
TEST_EPOCHS = 1
LEARNING_RATE = 2e-4

TRAIN_RATIO = 0.9
RANDOM_SEED = 42
SEED = RANDOM_SEED

USE_AMP = True

MSE_WEIGHT = 1.0
LPIPS_WEIGHT = 0.05
COLOR_WEIGHT = 0.5
LPIPS_NET = "alex"

LATENT_DTYPE_NAME = "uint8"

# Width multiplier selector: 0.5 or 1.0.
WIDTH_SELECT = 0.5

# Latent size selector: 16, 32, or 64.
# 16 -> 16x16x8 (approx. 384x), 32 -> 32x32x16 (approx. 48x),
# 64 -> 64x64x32 (approx. 6x). Shape notation is HxWxC.
LATENT_CONFIG = 32

LATENT_CONFIGS = {
    16: {"latent_spatial_size": 16, "latent_channels": 8},
    32: {"latent_spatial_size": 32, "latent_channels": 16},
    64: {"latent_spatial_size": 64, "latent_channels": 32},
}

# Single MobileNetV2 configuration for this run.
MODEL_CONFIGS = [
    {"width_mult": WIDTH_SELECT, **LATENT_CONFIGS[LATENT_CONFIG]},
]


# ============================================================
# ACTIVE MODEL SETTINGS
# Updated before each model is created.
# ============================================================

WIDTH_MULT = 0.5
LATENT_SPATIAL_SIZE = 16
LATENT_CHANNELS = 8
LATENT_WIDTH = LATENT_SPATIAL_SIZE
LATENT_HEIGHT = LATENT_SPATIAL_SIZE
TARGET_OUTPUT_STRIDE = IMAGE_WIDTH // LATENT_SPATIAL_SIZE

MODEL_FOLDER_NAME = ""
LOCAL_MODEL_ROOT = OUTPUT_DIR
OUTPUT_MODEL_ZIP = OUTPUT_DIR / "model.zip"
ENCODER_DIR = LOCAL_MODEL_ROOT / "encoder"
DECODER_DIR = LOCAL_MODEL_ROOT / "decoder"
FULL_DIR = LOCAL_MODEL_ROOT / "full"
LATEST_FULL_CHECKPOINT = FULL_DIR / "latest_full_checkpoint.pth"
CONFIG_PATH = LOCAL_MODEL_ROOT / "training_config.json"
METRICS_CSV_PATH = LOCAL_MODEL_ROOT / "metrics.csv"


def is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def validate_settings():
    if IMAGE_WIDTH != IMAGE_HEIGHT:
        raise ValueError("IMAGE_WIDTH and IMAGE_HEIGHT must be equal.")

    if IMAGE_WIDTH % LATENT_SPATIAL_SIZE != 0:
        raise ValueError("IMAGE_WIDTH must be divisible by LATENT_SPATIAL_SIZE.")

    ratio = IMAGE_WIDTH // LATENT_SPATIAL_SIZE

    if not is_power_of_two(ratio):
        raise ValueError(
            f"IMAGE_WIDTH / LATENT_SPATIAL_SIZE must be power of two. Got {ratio}."
        )

    if ratio not in (8, 16, 32):
        raise ValueError(
            "MobileNetV2 configurations in this file support latent sizes "
            "64, 32, and 16 for a 512x512 input."
        )

    if WIDTH_MULT <= 0:
        raise ValueError("WIDTH_MULT must be greater than zero.")

    return ratio


def width_tag(width_mult):
    return str(width_mult).replace(".", "p")


def activate_config(config):
    global WIDTH_MULT
    global LATENT_SPATIAL_SIZE
    global LATENT_CHANNELS
    global LATENT_WIDTH
    global LATENT_HEIGHT
    global TARGET_OUTPUT_STRIDE
    global MODEL_FOLDER_NAME
    global LOCAL_MODEL_ROOT
    global OUTPUT_MODEL_ZIP
    global ENCODER_DIR
    global DECODER_DIR
    global FULL_DIR
    global LATEST_FULL_CHECKPOINT
    global CONFIG_PATH
    global METRICS_CSV_PATH

    WIDTH_MULT = float(config["width_mult"])
    LATENT_SPATIAL_SIZE = int(config["latent_spatial_size"])
    LATENT_CHANNELS = int(config["latent_channels"])
    LATENT_WIDTH = LATENT_SPATIAL_SIZE
    LATENT_HEIGHT = LATENT_SPATIAL_SIZE
    TARGET_OUTPUT_STRIDE = validate_settings()

    MODEL_FOLDER_NAME = (
        f"mobilenetv2_w{width_tag(WIDTH_MULT)}_latent_"
        f"{LATENT_SPATIAL_SIZE}x{LATENT_SPATIAL_SIZE}x{LATENT_CHANNELS}_"
        f"{LATENT_DTYPE_NAME}_lpips"
    )

    LOCAL_MODEL_ROOT = OUTPUT_DIR / MODEL_FOLDER_NAME
    OUTPUT_MODEL_ZIP = OUTPUT_DIR / f"{MODEL_FOLDER_NAME}.zip"

    ENCODER_DIR = LOCAL_MODEL_ROOT / "encoder"
    DECODER_DIR = LOCAL_MODEL_ROOT / "decoder"
    FULL_DIR = LOCAL_MODEL_ROOT / "full"

    LATEST_FULL_CHECKPOINT = FULL_DIR / "latest_full_checkpoint.pth"
    CONFIG_PATH = LOCAL_MODEL_ROOT / "training_config.json"
    METRICS_CSV_PATH = LOCAL_MODEL_ROOT / "metrics.csv"


def config_label(config):
    size = config["latent_spatial_size"]
    channels = config["latent_channels"]
    return (
        f"MobileNetV2 width={config['width_mult']} - "
        f"{size}x{size}x{channels}"
    )


# ============================================================
# DATASET
# ============================================================

class RawRGBDataset(Dataset):
    def __init__(self, raw_dir: Path):
        all_files = sorted(raw_dir.glob("*.rgb"))

        if len(all_files) == 0:
            raise RuntimeError(f"No .rgb files found in: {raw_dir}")

        # Use the first MAX_IMAGES files only.
        files = all_files[:MAX_IMAGES] if MAX_IMAGES else all_files

        self.expected_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS

        # Preload all images into RAM as uint8 bytes (approx. 1.6 GB for
        # 2000 images) so no disk reads happen during training.
        self.buffers = []
        for file_path in files:
            raw_bytes = file_path.read_bytes()

            if len(raw_bytes) != self.expected_size:
                raise RuntimeError(
                    f"Wrong raw size: {file_path}, got {len(raw_bytes)}, "
                    f"expected {self.expected_size}"
                )

            self.buffers.append(raw_bytes)

        print(
            f"Preloaded {len(self.buffers)} images into RAM "
            f"({sum(len(item) for item in self.buffers) / 1e6:.1f} MB)."
        )

    def __len__(self):
        return len(self.buffers)

    def __getitem__(self, index):
        array = np.frombuffer(self.buffers[index], dtype=np.uint8)
        array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32) / 255.0

        return tensor


def build_data_loaders(dataset):
    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training loader is empty. Dataset must contain at least one full batch."
        )

    if len(val_loader) == 0:
        raise RuntimeError("Validation loader is empty.")

    return train_dataset, val_dataset, train_loader, val_loader


# ============================================================
# MOBILENETV2 MODEL
# ============================================================

def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor

    new_value = max(
        min_value,
        int(value + divisor / 2) // divisor * divisor,
    )

    if new_value < 0.9 * value:
        new_value += divisor

    return int(new_value)


class ConvBNReLU6(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
    ):
        padding = (kernel_size - 1) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("InvertedResidual stride must be 1 or 2.")

        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBNReLU6(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                )
            )

        layers.extend([
            ConvBNReLU6(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            ),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)

        return self.conv(x)


# Standard MobileNetV2 stage definition: expansion, channels, repeats, stride.
MOBILENETV2_STAGE_SETTINGS = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


class MobileNetV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        stem_channels = make_divisible(32 * WIDTH_MULT, 8)
        layers = [
            ConvBNReLU6(
                IMAGE_CHANNELS,
                stem_channels,
                kernel_size=3,
                stride=2,
            )
        ]

        current_channels = stem_channels
        current_stride = 2
        stage_records = []

        for expand_ratio, base_channels, repeats, stage_stride in MOBILENETV2_STAGE_SETTINGS:
            if current_stride * stage_stride > TARGET_OUTPUT_STRIDE:
                break

            output_channels = make_divisible(base_channels * WIDTH_MULT, 8)
            stage_input_channels = current_channels

            for block_index in range(repeats):
                stride = stage_stride if block_index == 0 else 1
                layers.append(
                    InvertedResidual(
                        current_channels,
                        output_channels,
                        stride=stride,
                        expand_ratio=expand_ratio,
                    )
                )
                current_channels = output_channels

            current_stride *= stage_stride
            stage_records.append(
                {
                    "input_channels": stage_input_channels,
                    "output_channels": output_channels,
                    "repeats": repeats,
                    "stride": stage_stride,
                    "expand_ratio": expand_ratio,
                }
            )

        if current_stride != TARGET_OUTPUT_STRIDE:
            raise RuntimeError(
                f"Encoder output stride is {current_stride}, "
                f"expected {TARGET_OUTPUT_STRIDE}."
            )

        self.features = nn.Sequential(*layers)
        self.latent_projection = nn.Sequential(
            nn.Conv2d(
                current_channels,
                LATENT_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.feature_channels = current_channels
        self.stem_channels = stem_channels
        self.stage_records = stage_records

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x = self.features(x)
        return self.latent_projection(x)


class MobileNetV2DecoderStage(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        repeats,
        upsample,
        expand_ratio,
    ):
        super().__init__()

        same_channel_blocks = []

        for _ in range(max(0, repeats - 1)):
            same_channel_blocks.append(
                InvertedResidual(
                    input_channels,
                    input_channels,
                    stride=1,
                    expand_ratio=expand_ratio,
                )
            )

        self.same_channel_blocks = nn.Sequential(*same_channel_blocks)
        self.upsample = upsample
        self.transition = InvertedResidual(
            input_channels,
            output_channels,
            stride=1,
            expand_ratio=expand_ratio,
        )

    def forward(self, x):
        x = self.same_channel_blocks(x)

        if self.upsample:
            x = F.interpolate(
                x,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )

        return self.transition(x)


class MobileNetV2Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        current_channels = encoder.feature_channels

        self.input_projection = ConvBNReLU6(
            LATENT_CHANNELS,
            current_channels,
            kernel_size=3,
            stride=1,
        )

        decoder_stages = []

        for record in reversed(encoder.stage_records):
            decoder_stages.append(
                MobileNetV2DecoderStage(
                    input_channels=record["output_channels"],
                    output_channels=record["input_channels"],
                    repeats=record["repeats"],
                    upsample=record["stride"] == 2,
                    expand_ratio=record["expand_ratio"],
                )
            )
            current_channels = record["input_channels"]

        self.stages = nn.Sequential(*decoder_stages)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                current_channels,
                IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = z.to(dtype=torch.float32)
        x = self.input_projection(z)
        x = self.stages(x)
        return self.output_layer(x)


class MobileNetV2Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = MobileNetV2Encoder()
        self.decoder = MobileNetV2Decoder(self.encoder)

    def fake_quantize_latent_float32(self, z):
        z_scaled = z * 255.0
        z_rounded = torch.round(z_scaled)

        # Straight-through estimator.
        z_quantized = z_scaled + (z_rounded - z_scaled).detach()
        z_dequantized = z_quantized / 255.0

        return z_dequantized.to(dtype=torch.float32)

    def forward(self, x):
        x = x.to(dtype=torch.float32)

        z = self.encoder(x)
        z_q = self.fake_quantize_latent_float32(z)
        x_hat = self.decoder(z_q)

        return x_hat, z_q


class EncoderDeploy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        z = self.encoder(x)

        z_q = torch.round(z * 255.0)
        z_q = torch.clamp(z_q, 0.0, 255.0)

        return z_q.to(dtype=torch.uint8)


class DecoderDeploy(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        z = z.to(dtype=torch.uint8).to(dtype=torch.float32) / 255.0
        x_hat = self.decoder(z)

        return x_hat.to(dtype=torch.float32)


# ============================================================
# PERCEPTION LOSS
# ============================================================

class PerceptualReconstructionLoss(nn.Module):
    def __init__(self, perceptual_model):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.perceptual_model = perceptual_model

    def forward(self, x_hat, target):
        x_hat = x_hat.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        mse_loss = self.mse(x_hat, target)

        # LPIPS expects images in [-1, 1].
        x_hat_lpips = (x_hat * 2.0 - 1.0).to(dtype=torch.float32)
        target_lpips = (target * 2.0 - 1.0).to(dtype=torch.float32)

        lpips_loss = self.perceptual_model(x_hat_lpips, target_lpips).mean()
        lpips_loss = lpips_loss.to(dtype=torch.float32)

        # Keeps RGB balance and reduces gray/desaturated output.
        color_hat = x_hat.mean(dim=[2, 3])
        color_target = target.mean(dim=[2, 3])
        color_loss = self.l1(color_hat, color_target)

        total_loss = (
            MSE_WEIGHT * mse_loss
            + LPIPS_WEIGHT * lpips_loss
            + COLOR_WEIGHT * color_loss
        )

        return total_loss, mse_loss, lpips_loss, color_loss


def create_perceptual_criterion(device):
    perceptual_model = lpips.LPIPS(net=LPIPS_NET).to(device)
    perceptual_model.eval()

    for parameter in perceptual_model.parameters():
        parameter.requires_grad = False

    criterion = PerceptualReconstructionLoss(perceptual_model)
    return criterion, perceptual_model


# ============================================================
# METRICS
# ============================================================

def psnr_from_mse(mse_value):
    if mse_value <= 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse_value)


def latent_size_bytes_uint8():
    return LATENT_SPATIAL_SIZE * LATENT_SPATIAL_SIZE * LATENT_CHANNELS


def input_size_bytes_uint8():
    return IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def make_dirs():
    # Fresh run: delete stale epoch files from previous runs so the ZIP
    # never mixes old and new models. (metrics.csv, configs and the latest
    # checkpoint are overwritten anyway.)
    for stale_dir in (ENCODER_DIR, DECODER_DIR):
        if stale_dir.exists():
            print("Removing stale dir:", stale_dir)
            shutil.rmtree(stale_dir)

    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    DECODER_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_training_config():
    config = {
        "model_name": "mobilenetv2_latent_autoencoder_uint8_lpips",
        "model_folder_name": MODEL_FOLDER_NAME,
        "architecture": "MobileNetV2 encoder + mirrored MobileNetV2 decoder",
        "width_mult": WIDTH_MULT,
        "pretrained": False,

        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "image_channels": IMAGE_CHANNELS,

        "latent_spatial_size": LATENT_SPATIAL_SIZE,
        "latent_width": LATENT_WIDTH,
        "latent_height": LATENT_HEIGHT,
        "latent_channels": LATENT_CHANNELS,
        "latent_dtype": "uint8",
        "target_output_stride": TARGET_OUTPUT_STRIDE,

        "latent_size_bytes": latent_size_bytes_uint8(),
        "input_size_bytes": input_size_bytes_uint8(),
        "compression_ratio": input_size_bytes_uint8() / latent_size_bytes_uint8(),

        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "use_amp": USE_AMP,

        "loss": "MSE + LPIPS + color loss",
        "mse_weight": MSE_WEIGHT,
        "lpips_weight": LPIPS_WEIGHT,
        "color_weight": COLOR_WEIGHT,
        "lpips_net": LPIPS_NET,
    }

    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=2)

    print("Saved config:", CONFIG_PATH)


def init_metrics_csv():
    with open(METRICS_CSV_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_total_loss",
            "train_mse_loss",
            "train_lpips_loss",
            "train_color_loss",
            "train_psnr",
            "val_total_loss",
            "val_mse_loss",
            "val_lpips_loss",
            "val_color_loss",
            "val_psnr",
            "val_loss_delta",
            "train_val_gap",
            "is_best",
            "latent_kb_uint8",
            "compression_ratio",
            "epoch_time_seconds",
        ])

    print("Created metrics CSV:", METRICS_CSV_PATH)


def append_metrics_csv(
    epoch,
    train_total,
    train_mse,
    train_lpips,
    train_color,
    train_psnr,
    val_total,
    val_mse,
    val_lpips,
    val_color,
    val_psnr,
    val_loss_delta,
    train_val_gap,
    is_best,
    epoch_time,
):
    with open(METRICS_CSV_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,
            train_total,
            train_mse,
            train_lpips,
            train_color,
            train_psnr,
            val_total,
            val_mse,
            val_lpips,
            val_color,
            val_psnr,
            val_loss_delta,
            train_val_gap,
            is_best,
            latent_size_bytes_uint8() / 1024,
            input_size_bytes_uint8() / latent_size_bytes_uint8(),
            epoch_time,
        ])


def save_latest_full_checkpoint(
    model,
    optimizer,
    epoch,
    train_total,
    train_mse,
    train_lpips,
    train_color,
    train_psnr,
    val_total,
    val_mse,
    val_lpips,
    val_color,
    val_psnr,
):
    torch.save(
        {
            "model_name": "mobilenetv2_latent_autoencoder_uint8_lpips",
            "model_folder_name": MODEL_FOLDER_NAME,
            "architecture": "MobileNetV2 encoder + mirrored MobileNetV2 decoder",
            "width_mult": WIDTH_MULT,
            "pretrained": False,
            "epoch": epoch,

            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "image_channels": IMAGE_CHANNELS,

            "latent_spatial_size": LATENT_SPATIAL_SIZE,
            "latent_width": LATENT_WIDTH,
            "latent_height": LATENT_HEIGHT,
            "latent_channels": LATENT_CHANNELS,
            "latent_dtype": "uint8",
            "target_output_stride": TARGET_OUTPUT_STRIDE,

            "latent_size_bytes": latent_size_bytes_uint8(),
            "input_size_bytes": input_size_bytes_uint8(),
            "compression_ratio": input_size_bytes_uint8() / latent_size_bytes_uint8(),

            "loss": "MSE + LPIPS + color loss",
            "mse_weight": MSE_WEIGHT,
            "lpips_weight": LPIPS_WEIGHT,
            "color_weight": COLOR_WEIGHT,
            "lpips_net": LPIPS_NET,

            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "train_total_loss": train_total,
            "train_mse_loss": train_mse,
            "train_lpips_loss": train_lpips,
            "train_color_loss": train_color,
            "train_psnr": train_psnr,

            "val_total_loss": val_total,
            "val_mse_loss": val_mse,
            "val_lpips_loss": val_lpips,
            "val_color_loss": val_color,
            "val_psnr": val_psnr,
        },
        LATEST_FULL_CHECKPOINT,
    )

    print("Saved latest full:", LATEST_FULL_CHECKPOINT)


def save_deploy_models(model, epoch, device):
    model.eval()

    encoder_deploy = EncoderDeploy(model.encoder).to(device).eval()
    decoder_deploy = DecoderDeploy(model.decoder).to(device).eval()

    example_image = torch.zeros(
        1,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        dtype=torch.float32,
        device=device,
    )

    example_latent = torch.zeros(
        1,
        LATENT_CHANNELS,
        LATENT_SPATIAL_SIZE,
        LATENT_SPATIAL_SIZE,
        dtype=torch.uint8,
        device=device,
    )

    encoder_script = torch.jit.trace(encoder_deploy, example_image)
    decoder_script = torch.jit.trace(decoder_deploy, example_latent)

    encoder_path = ENCODER_DIR / f"{MODEL_FOLDER_NAME}_epoch_{epoch:03d}_encoder.pt"
    decoder_path = DECODER_DIR / f"{MODEL_FOLDER_NAME}_epoch_{epoch:03d}_decoder.pt"

    encoder_script.save(str(encoder_path))
    decoder_script.save(str(decoder_path))

    print("Saved encoder:", encoder_path)
    print("Saved decoder:", decoder_path)


def zip_and_save_models(best_epoch):
    print()
    print("Creating model zip...")

    if OUTPUT_MODEL_ZIP.exists():
        OUTPUT_MODEL_ZIP.unlink()

    local_zip_base = LOCAL_MODEL_ROOT
    local_zip_path = Path(str(local_zip_base) + ".zip")

    if local_zip_path.exists():
        local_zip_path.unlink()

    shutil.make_archive(
        base_name=str(local_zip_base),
        format="zip",
        root_dir=str(LOCAL_MODEL_ROOT.parent),
        base_dir=LOCAL_MODEL_ROOT.name,
    )

    shutil.copy2(local_zip_path, OUTPUT_MODEL_ZIP)

    print("Local zip:", local_zip_path)
    print("Output zip:", OUTPUT_MODEL_ZIP)
    print("Output zip size MB:", round(OUTPUT_MODEL_ZIP.stat().st_size / 1e6, 2))


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    total_sum = 0.0
    mse_sum = 0.0
    lpips_sum = 0.0
    color_sum = 0.0

    for batch in loader:
        batch = batch.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                x_hat, _ = model(batch)
                x_hat = x_hat.to(dtype=torch.float32)
                total_loss, mse_loss, lpips_loss, color_loss = criterion(x_hat, batch)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x_hat, _ = model(batch)
            x_hat = x_hat.to(dtype=torch.float32)
            total_loss, mse_loss, lpips_loss, color_loss = criterion(x_hat, batch)

            total_loss.backward()
            optimizer.step()

        total_sum += float(total_loss.item())
        mse_sum += float(mse_loss.item())
        lpips_sum += float(lpips_loss.item())
        color_sum += float(color_loss.item())

    n = len(loader)

    avg_total = total_sum / n
    avg_mse = mse_sum / n
    avg_lpips = lpips_sum / n
    avg_color = color_sum / n
    avg_psnr = psnr_from_mse(avg_mse)

    return avg_total, avg_mse, avg_lpips, avg_color, avg_psnr


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_sum = 0.0
    mse_sum = 0.0
    lpips_sum = 0.0
    color_sum = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device, dtype=torch.float32, non_blocking=True)
        batch_size = batch.size(0)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                x_hat, _ = model(batch)
                x_hat = x_hat.to(dtype=torch.float32)
                total_loss, mse_loss, lpips_loss, color_loss = criterion(x_hat, batch)
        else:
            x_hat, _ = model(batch)
            x_hat = x_hat.to(dtype=torch.float32)
            total_loss, mse_loss, lpips_loss, color_loss = criterion(x_hat, batch)

        # Weight by batch size: the last batch may be smaller (drop_last=False).
        total_sum += float(total_loss.item()) * batch_size
        mse_sum += float(mse_loss.item()) * batch_size
        lpips_sum += float(lpips_loss.item()) * batch_size
        color_sum += float(color_loss.item()) * batch_size
        total_samples += batch_size

    avg_total = total_sum / total_samples
    avg_mse = mse_sum / total_samples
    avg_lpips = lpips_sum / total_samples
    avg_color = color_sum / total_samples
    avg_psnr = psnr_from_mse(avg_mse)

    return avg_total, avg_mse, avg_lpips, avg_color, avg_psnr


# ============================================================
# MODEL HELPERS
# ============================================================

def seed_everything():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def clear_model_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_model_shapes(model, device):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        test_x = torch.zeros(
            1,
            IMAGE_CHANNELS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            dtype=torch.float32,
            device=device,
        )
        test_z = model.encoder(test_x)
        test_output = model.decoder(test_z)

    if was_training:
        model.train()

    expected_latent_shape = (
        1,
        LATENT_CHANNELS,
        LATENT_SPATIAL_SIZE,
        LATENT_SPATIAL_SIZE,
    )
    expected_output_shape = (
        1,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    print("Real latent shape:", test_z.shape)
    print("Real output shape:", test_output.shape)

    if tuple(test_z.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Wrong latent shape. Got {tuple(test_z.shape)}, "
            f"expected {expected_latent_shape}"
        )

    if tuple(test_output.shape) != expected_output_shape:
        raise RuntimeError(
            f"Wrong decoder output shape. Got {tuple(test_output.shape)}, "
            f"expected {expected_output_shape}"
        )


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def print_model_info(model, device, epochs):
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print("Dataset:", DATASET_DIR)
    print("Max images:", MAX_IMAGES)
    print("Batch size:", BATCH_SIZE)
    print("Workers:", NUM_WORKERS)
    print("Epochs:", epochs)
    print("Learning rate:", LEARNING_RATE)

    print("Architecture:", "MobileNetV2 autoencoder")
    print("Width multiplier:", WIDTH_MULT)
    print("Pretrained:", False)
    print("Image size:", f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print("Latent shape:", f"{LATENT_CHANNELS}x{LATENT_SPATIAL_SIZE}x{LATENT_SPATIAL_SIZE}")
    print("Target output stride:", TARGET_OUTPUT_STRIDE)
    print("Latent dtype:", LATENT_DTYPE_NAME)

    print("Encoder parameters:", count_parameters(model.encoder))
    print("Decoder parameters:", count_parameters(model.decoder))
    print("Total parameters:", count_parameters(model))

    print("Input size bytes:", input_size_bytes_uint8())
    print("Latent size bytes:", latent_size_bytes_uint8())
    print("Latent size KB:", round(latent_size_bytes_uint8() / 1024, 2))
    print(
        "Compression ratio:",
        round(input_size_bytes_uint8() / latent_size_bytes_uint8(), 3),
        "x",
    )

    print(
        "Loss:",
        f"{MSE_WEIGHT} * MSE + {LPIPS_WEIGHT} * LPIPS + "
        f"{COLOR_WEIGHT} * color",
    )
    print("LPIPS net:", LPIPS_NET)
    print("Model folder name:", MODEL_FOLDER_NAME)
    print("Output model zip:", OUTPUT_MODEL_ZIP)


# ============================================================
# ONE-EPOCH TESTS
# ============================================================

def test_one_config(config, train_loader, val_loader, device):
    activate_config(config)
    seed_everything()

    label = config_label(config)
    print()
    print("=" * 70)
    print(f"ONE-EPOCH TEST: {label}")
    print("=" * 70)

    model = MobileNetV2Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    criterion = None
    perceptual_model = None

    try:
        check_model_shapes(model, device)
        criterion, perceptual_model = create_perceptual_criterion(device)

        for test_epoch in range(1, TEST_EPOCHS + 1):
            train_values = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            val_values = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Test epoch {test_epoch}/{TEST_EPOCHS} passed - "
                f"train loss: {train_values[0]:.6f}, "
                f"val loss: {val_values[0]:.6f}"
            )

        print(f"TEST PASSED: {label}")
    finally:
        del model
        del optimizer
        del scaler

        if criterion is not None:
            del criterion

        if perceptual_model is not None:
            del perceptual_model

        clear_model_memory()


def run_all_tests(train_loader, val_loader, device):
    print()
    print("#" * 70)
    print("STARTING ONE-EPOCH TEST FOR THE MODEL")
    print("#" * 70)

    for config in MODEL_CONFIGS:
        test_one_config(config, train_loader, val_loader, device)

    print()
    print("#" * 70)
    print("ONE-EPOCH TEST PASSED.")
    print("REAL 500-EPOCH TRAINING WILL NOW START FROM SCRATCH.")
    print("#" * 70)


# ============================================================
# REAL TRAINING
# ============================================================

def train_one_config(config, train_loader, val_loader, device):
    activate_config(config)
    seed_everything()

    label = config_label(config)
    print()
    print("=" * 70)
    print(f"REAL TRAINING: {label}")
    print("=" * 70)

    make_dirs()
    save_training_config()
    init_metrics_csv()

    model = MobileNetV2Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    criterion = None
    perceptual_model = None

    try:
        check_model_shapes(model, device)
        print_model_info(model, device, EPOCHS)

        criterion, perceptual_model = create_perceptual_criterion(device)

        prev_val_total = None
        prev_val_psnr = None
        best_val_total = float("inf")
        best_epoch = 0
        best_state = None
        best_metrics = None

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()

            train_total, train_mse, train_lpips, train_color, train_psnr = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                )
            )

            val_total, val_mse, val_lpips, val_color, val_psnr = (
                validate_one_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                )
            )

            epoch_time = time.time() - epoch_start

            # Convergence: change in validation loss vs previous epoch,
            # plus the train/validation loss gap.
            val_loss_delta = "" if prev_val_total is None else val_total - prev_val_total
            train_val_gap = val_total - train_total

            if prev_val_total is None or prev_val_total == 0:
                loss_improve = "-"
            else:
                loss_improve = f"{(prev_val_total - val_total) / prev_val_total * 100:+.2f}%"

            if prev_val_psnr is None or prev_val_psnr == 0:
                psnr_improve = "-"
            else:
                psnr_improve = f"{(val_psnr - prev_val_psnr) / prev_val_psnr * 100:+.2f}%"

            prev_val_total = val_total
            prev_val_psnr = val_psnr

            # Best model tracking on validation total loss.
            if val_total < best_val_total:
                best_val_total = val_total
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_metrics = (
                    train_total, train_mse, train_lpips, train_color, train_psnr,
                    val_total, val_mse, val_lpips, val_color, val_psnr,
                )
                is_best = 1
            else:
                is_best = 0

            print()
            print(f"mb_w{width_tag(WIDTH_MULT)}_{LATENT_SPATIAL_SIZE} e{epoch}")
            print(f"total loss: {val_total:.6f} ({loss_improve})")
            print(f"psnr: {val_psnr:.3f} dB ({psnr_improve})")
            print(f"epoch time: {epoch_time:.1f} s")
            print(f"Best so far: epoch {best_epoch} val {best_val_total:.6f}")

            append_metrics_csv(
                epoch=epoch,
                train_total=train_total,
                train_mse=train_mse,
                train_lpips=train_lpips,
                train_color=train_color,
                train_psnr=train_psnr,
                val_total=val_total,
                val_mse=val_mse,
                val_lpips=val_lpips,
                val_color=val_color,
                val_psnr=val_psnr,
                val_loss_delta=val_loss_delta,
                train_val_gap=train_val_gap,
                is_best=is_best,
                epoch_time=epoch_time,
            )

            if epoch == EPOCHS:
                print()
                print(f"Loading best weights from epoch {best_epoch}...")
                model.load_state_dict(best_state)

                (
                    best_train_total, best_train_mse, best_train_lpips,
                    best_train_color, best_train_psnr, best_val_total_end,
                    best_val_mse, best_val_lpips, best_val_color, best_val_psnr,
                ) = best_metrics

                save_latest_full_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=best_epoch,
                    train_total=best_train_total,
                    train_mse=best_train_mse,
                    train_lpips=best_train_lpips,
                    train_color=best_train_color,
                    train_psnr=best_train_psnr,
                    val_total=best_val_total_end,
                    val_mse=best_val_mse,
                    val_lpips=best_val_lpips,
                    val_color=best_val_color,
                    val_psnr=best_val_psnr,
                )
                save_deploy_models(model=model, epoch=best_epoch, device=device)

        zip_and_save_models(best_epoch)

        print()
        print("Training finished.")
        print("Latest full checkpoint:", LATEST_FULL_CHECKPOINT)
        print("Final output zip:", OUTPUT_MODEL_ZIP)
    finally:
        del model
        del optimizer
        del scaler

        if criterion is not None:
            del criterion

        if perceptual_model is not None:
            del perceptual_model

        clear_model_memory()


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        # Fixed 512x512 input: autotune convolution algorithms once.
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark: True")

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print("Dataset:", DATASET_DIR)
    print("Max images:", MAX_IMAGES)
    print("Real epochs per model:", EPOCHS)
    print("Test epochs per model:", TEST_EPOCHS)
    print("Number of models:", len(MODEL_CONFIGS))

    dataset = RawRGBDataset(DATASET_DIR)
    train_dataset, val_dataset, train_loader, val_loader = build_data_loaders(dataset)

    print("Total samples:", len(dataset))
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # Real output is not written unless the one-epoch test passes.
    run_all_tests(train_loader, val_loader, device)

    for config in MODEL_CONFIGS:
        train_one_config(config, train_loader, val_loader, device)

    print()
    print("#" * 70)
    print("MODEL FINISHED SUCCESSFULLY.")
    print("Output model directory:", OUTPUT_DIR)
    print("#" * 70)


if __name__ == "__main__":
    main()
