"""Cheng2020 Anchor fine-tuning with rate-distortion loss."""

from pathlib import Path
import os
import time
import math
import random
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from compressai.zoo import cheng2020_anchor


# ============================================================
# PARAMETERS - SET THESE BEFORE RUNNING
# ============================================================

# Local dataset folder holding 512x512x3 raw RGB files (*.rgb).
YOUR_DATASET_DIR = Path("./dataset_raw")

# Model export folder (checkpoints, models, metrics, ZIPs).
YOUR_MODEL_EXPORT_DIR = Path("./output")

DATASET_DIR = YOUR_DATASET_DIR
OUTPUT_DIR = YOUR_MODEL_EXPORT_DIR

MAX_IMAGES = 2000

USE_PRETRAINED = True

# Quality selector: 3 or 6 (official CompressAI MSE lambdas).
QUALITY_SELECT = 6

QUALITY_LAMBDAS = {3: 0.0067, 6: 0.0483}

QUALITY = QUALITY_SELECT
LAMBDA_RD = QUALITY_LAMBDAS[QUALITY_SELECT]

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANNELS = 3

BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 50
TEST_EPOCHS = 1
LEARNING_RATE = 5e-6
AUX_LEARNING_RATE = 1e-5
CALIB_EPOCHS = 50

TRAIN_RATIO = 0.9
RANDOM_SEED = 42
SEED = RANDOM_SEED

USE_AMP = True

LOCAL_MODEL_ROOT = OUTPUT_DIR / f"cheng_q{QUALITY}_anchor"
OUTPUT_MODEL_ZIP = OUTPUT_DIR / f"cheng_q{QUALITY}_anchor.zip"

# ============================================================
# OUTPUT PATHS
# ============================================================

ENCODER_DIR = LOCAL_MODEL_ROOT / "encoder"
DECODER_DIR = LOCAL_MODEL_ROOT / "decoder"
FULL_DIR = LOCAL_MODEL_ROOT / "full"

LATEST_FULL_CHECKPOINT = FULL_DIR / "latest_full_checkpoint.pth"


# ============================================================
# DATASET
# ============================================================

class RawRGBDataset(Dataset):
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
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
                    f"Wrong raw size: {file_path}, got {len(raw_bytes)}, expected {self.expected_size}"
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

        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0

        return tensor


# ============================================================
# LOSS AND METRICS
# ============================================================

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda: float):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        batch_size, _, height, width = target.size()
        num_pixels = batch_size * height * width

        bpp_loss = sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values()
        )

        mse_loss = self.mse(output["x_hat"], target)

        loss = bpp_loss + self.lmbda * 255.0 ** 2 * mse_loss

        return loss, bpp_loss, mse_loss


def get_aux_parameters(model):
    """Entropy-bottleneck quantile tables, trained by a separate optimizer."""
    return [
        param
        for param_name, param in model.named_parameters()
        if param_name.endswith(".quantiles")
    ]


def load_quantile_tables(model, tables):
    """Restore quantile tables without load_state_dict (partial dicts break
    CompressAI buffer handling even with strict=False)."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in tables.items():
            params[key].copy_(value)


def get_main_parameters(model):
    """All parameters except the quantile tables (RD loss only)."""
    return [
        param
        for param_name, param in model.named_parameters()
        if not param_name.endswith(".quantiles")
    ]


def psnr_from_mse(mse_value: float) -> float:
    if mse_value <= 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse_value)


METRICS_CSV_PATH = LOCAL_MODEL_ROOT / "metrics.csv"
CONFIG_PATH = LOCAL_MODEL_ROOT / "training_config.json"


def save_training_config():
    import json as json_module

    config = {
        "quality": QUALITY,
        "lambda_rd": LAMBDA_RD,
        "use_pretrained": USE_PRETRAINED,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "image_channels": IMAGE_CHANNELS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "aux_learning_rate": AUX_LEARNING_RATE,
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "use_amp": USE_AMP,
    }

    with open(CONFIG_PATH, "w") as file:
        json_module.dump(config, file, indent=2)

    print("Saved config:", CONFIG_PATH)


def init_metrics_csv():
    import csv as csv_module

    with open(METRICS_CSV_PATH, "w", newline="") as file:
        writer = csv_module.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_bpp",
            "train_aux_loss",
            "train_rd_loss",
            "train_psnr",
            "val_loss",
            "val_bpp",
            "val_aux_loss",
            "val_rd_loss",
            "val_psnr",
            "val_loss_delta",
            "train_val_gap",
            "is_best",
            "epoch_time_seconds",
        ])

    print("Created metrics CSV:", METRICS_CSV_PATH)


def append_metrics_csv(
    epoch,
    train_loss,
    train_bpp,
    train_aux_loss,
    train_rd_loss,
    train_psnr,
    val_loss,
    val_bpp,
    val_aux_loss,
    val_rd_loss,
    val_psnr,
    val_loss_delta,
    train_val_gap,
    is_best,
    epoch_time,
):
    import csv as csv_module

    with open(METRICS_CSV_PATH, "a", newline="") as file:
        writer = csv_module.writer(file)
        writer.writerow([
            epoch,
            train_loss,
            train_bpp,
            train_aux_loss,
            train_rd_loss,
            train_psnr,
            val_loss,
            val_bpp,
            val_aux_loss,
            val_rd_loss,
            val_psnr,
            val_loss_delta,
            train_val_gap,
            is_best,
            epoch_time,
        ])


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


def split_encoder_decoder_state(model):
    state = model.state_dict()

    encoder_state = {}
    decoder_state = {}

    for key, value in state.items():
        value_cpu = value.detach().cpu()

        # Drone side: everything needed for compress(), excluding final image decoder g_s.
        if not key.startswith("g_s."):
            encoder_state[key] = value_cpu

        # Ground side: everything needed for decompress(), excluding image analysis encoder g_a and hyper encoder h_a.
        if not key.startswith("g_a.") and not key.startswith("h_a."):
            decoder_state[key] = value_cpu

    return encoder_state, decoder_state


def save_deploy_models(model, epoch):
    model.update(force=True)

    encoder_state, decoder_state = split_encoder_decoder_state(model)

    encoder_path = ENCODER_DIR / f"cheng_q{QUALITY}_anchor_epoch_{epoch:03d}_drone_encoder_entropy.pth"
    decoder_path = DECODER_DIR / f"cheng_q{QUALITY}_anchor_epoch_{epoch:03d}_ground_decoder_entropy.pth"

    torch.save(
        {
            "model_name": "cheng2020_anchor",
            "quality": QUALITY,
            "epoch": epoch,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "state_dict": encoder_state,
        },
        encoder_path,
    )

    torch.save(
        {
            "model_name": "cheng2020_anchor",
            "quality": QUALITY,
            "epoch": epoch,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "state_dict": decoder_state,
        },
        decoder_path,
    )

    print("Saved encoder:", encoder_path)
    print("Saved decoder:", decoder_path)


def save_latest_full_checkpoint(model, optimizer, aux_optimizer, epoch, train_loss, val_loss, val_bpp, val_psnr, aux_calibrated, calib_best_aux, calib_best_rd):
    model.update(force=True)

    torch.save(
        {
            "model_name": "cheng2020_anchor",
            "quality": QUALITY,
            "epoch": epoch,
            "use_pretrained": USE_PRETRAINED,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "aux_optimizer_state_dict": aux_optimizer.state_dict(),
            "aux_calibrated": aux_calibrated,
            "calib_best_aux": calib_best_aux,
            "calib_best_rd": calib_best_rd,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bpp": val_bpp,
            "val_psnr": val_psnr,
            "lambda_rd": LAMBDA_RD,
        },
        LATEST_FULL_CHECKPOINT,
    )

    print("Saved latest full:", LATEST_FULL_CHECKPOINT)


def zip_and_save_models(best_epoch):
    print()
    print("Creating model zip...")

    if OUTPUT_MODEL_ZIP.exists():
        print("Removing old output zip:", OUTPUT_MODEL_ZIP)
        OUTPUT_MODEL_ZIP.unlink()

    # The archive is created directly at its final path; no copy step.
    shutil.make_archive(
        base_name=str(OUTPUT_MODEL_ZIP.with_suffix("")),
        format="zip",
        root_dir=str(LOCAL_MODEL_ROOT.parent),
        base_dir=LOCAL_MODEL_ROOT.name,
    )

    print("Output zip:", OUTPUT_MODEL_ZIP)
    print("Output zip size MB:", round(OUTPUT_MODEL_ZIP.stat().st_size / 1e6, 2))


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, aux_optimizer, scaler, device):
    model.train()

    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_aux = 0.0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                output = model(batch)
                rd_loss, bpp_loss, mse_loss = criterion(output, batch)

            scaler.scale(rd_loss).backward()
            scaler.step(optimizer)

            aux_loss = model.aux_loss()

            scaler.scale(aux_loss).backward()
            scaler.step(aux_optimizer)
            scaler.update()

        else:
            output = model(batch)
            rd_loss, bpp_loss, mse_loss = criterion(output, batch)

            rd_loss.backward()
            optimizer.step()

            aux_loss = model.aux_loss()

            aux_loss.backward()
            aux_optimizer.step()

        loss = rd_loss + aux_loss

        total_loss += loss.item()
        total_bpp += bpp_loss.item()
        total_mse += mse_loss.item()
        total_aux += aux_loss.item()

    n = len(loader)

    avg_loss = total_loss / n
    avg_bpp = total_bpp / n
    avg_mse = total_mse / n
    avg_psnr = psnr_from_mse(avg_mse)
    avg_aux = total_aux / n

    return avg_loss, avg_bpp, avg_mse, avg_psnr, avg_aux, avg_loss - avg_aux


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_aux = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        batch_size = batch.size(0)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                output = model(batch)
                rd_loss, bpp_loss, mse_loss = criterion(output, batch)
        else:
            output = model(batch)
            rd_loss, bpp_loss, mse_loss = criterion(output, batch)

        # Aux loss in FP32 on both train and validation for comparability.
        aux_loss = model.aux_loss()
        loss = rd_loss + aux_loss

        # Weight by batch size: the last batch may be smaller (drop_last=False).
        total_loss += loss.item() * batch_size
        total_bpp += bpp_loss.item() * batch_size
        total_mse += mse_loss.item() * batch_size
        total_aux += aux_loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_bpp = total_bpp / total_samples
    avg_mse = total_mse / total_samples
    avg_psnr = psnr_from_mse(avg_mse)
    avg_aux = total_aux / total_samples

    return avg_loss, avg_bpp, avg_mse, avg_psnr, avg_aux, avg_loss - avg_aux


def set_quality(quality, lambda_rd):
    """Switch all quality-dependent paths for the next training run."""
    global QUALITY
    global LAMBDA_RD
    global LOCAL_MODEL_ROOT
    global OUTPUT_MODEL_ZIP
    global ENCODER_DIR
    global DECODER_DIR
    global FULL_DIR
    global LATEST_FULL_CHECKPOINT
    global METRICS_CSV_PATH
    global CONFIG_PATH

    QUALITY = quality
    LAMBDA_RD = lambda_rd

    LOCAL_MODEL_ROOT = OUTPUT_DIR / f"cheng_q{QUALITY}_anchor"
    OUTPUT_MODEL_ZIP = OUTPUT_DIR / f"cheng_q{QUALITY}_anchor.zip"

    ENCODER_DIR = LOCAL_MODEL_ROOT / "encoder"
    DECODER_DIR = LOCAL_MODEL_ROOT / "decoder"
    FULL_DIR = LOCAL_MODEL_ROOT / "full"

    LATEST_FULL_CHECKPOINT = FULL_DIR / "latest_full_checkpoint.pth"
    METRICS_CSV_PATH = LOCAL_MODEL_ROOT / "metrics.csv"
    CONFIG_PATH = LOCAL_MODEL_ROOT / "training_config.json"


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    for quality, lambda_rd in [(QUALITY_SELECT, QUALITY_LAMBDAS[QUALITY_SELECT])]:
        set_quality(quality, lambda_rd)
        print()
        print("#" * 70)
        print(f"QUALITY q{quality} - LAMBDA {lambda_rd}")
        print("#" * 70)
        make_dirs()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            # Fixed 512x512 input: autotune convolution algorithms once.
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark: True")

        print("Device:", device)
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
        print("Model: cheng2020_anchor")
        print("Quality:", QUALITY)
        print("Use pretrained:", USE_PRETRAINED)
        print("Batch size:", BATCH_SIZE)
        print("Workers:", NUM_WORKERS)
        print("Epochs:", EPOCHS)
        print("Learning rate:", LEARNING_RATE)
        print("Aux learning rate:", AUX_LEARNING_RATE)
        print("Calib epochs:", CALIB_EPOCHS)
        print("Lambda RD:", LAMBDA_RD)
        print("Dataset:", DATASET_DIR)
        print("Max images:", MAX_IMAGES)
        print("Local save root:", LOCAL_MODEL_ROOT)
        print("Output model zip:", OUTPUT_MODEL_ZIP)

        dataset = RawRGBDataset(DATASET_DIR)

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

        print("Total samples:", len(dataset))
        print("Train samples:", len(train_dataset))
        print("Val samples:", len(val_dataset))

        model = cheng2020_anchor(
            quality=QUALITY,
            pretrained=USE_PRETRAINED,
        ).to(device)

        criterion = RateDistortionLoss(LAMBDA_RD)

        optimizer = optim.Adam(
            get_main_parameters(model),
            lr=LEARNING_RATE,
        )

        aux_optimizer = optim.Adam(
            get_aux_parameters(model),
            lr=AUX_LEARNING_RATE,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        save_training_config()
        init_metrics_csv()

        print()
        print("=" * 70)
        print("ONE-EPOCH TEST (model is discarded afterwards)")
        print("=" * 70)

        test_model = cheng2020_anchor(
            quality=QUALITY,
            pretrained=USE_PRETRAINED,
        ).to(device)
        test_optimizer = optim.Adam(get_main_parameters(test_model), lr=LEARNING_RATE)
        test_aux_optimizer = optim.Adam(get_aux_parameters(test_model), lr=AUX_LEARNING_RATE)
        test_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        test_train = train_one_epoch(
            test_model, train_loader, criterion, test_optimizer, test_aux_optimizer, test_scaler, device
        )
        test_val = validate_one_epoch(
            test_model, val_loader, criterion, device
        )
        print(
            f"Test epoch passed - train loss: {test_train[0]:.6f}, "
            f"val loss: {test_val[0]:.6f}"
        )
        print("TEST PASSED")

        del test_model
        del test_optimizer
        del test_aux_optimizer
        del test_scaler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prev_val_loss = None
        prev_val_psnr = None
        best_val_rd = float("inf")
        best_epoch = 0
        best_state = None
        best_scores = None

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()

            train_loss, train_bpp, train_mse, train_psnr, train_aux, train_rd = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                aux_optimizer,
                scaler,
                device,
            )

            val_loss, val_bpp, val_mse, val_psnr, val_aux, val_rd = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            epoch_time = time.time() - epoch_start

            # Convergence: change in validation loss vs previous epoch,
            # plus the train/validation loss gap.
            val_loss_delta = "" if prev_val_loss is None else val_loss - prev_val_loss
            train_val_gap = val_loss - train_loss

            if prev_val_loss is None or prev_val_loss == 0:
                loss_improve = "-"
            else:
                loss_improve = f"{(prev_val_loss - val_loss) / prev_val_loss * 100:+.2f}%"

            if prev_val_psnr is None or prev_val_psnr == 0:
                psnr_improve = "-"
            else:
                psnr_improve = f"{(val_psnr - prev_val_psnr) / prev_val_psnr * 100:+.2f}%"

            prev_val_loss = val_loss
            prev_val_psnr = val_psnr

            # Best model tracking on validation RD loss (aux excluded).
            if val_rd < best_val_rd:
                best_val_rd = val_rd
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_scores = (
                    train_loss, train_bpp, train_psnr,
                    val_loss, val_bpp, val_psnr,
                )
                is_best = 1
            else:
                is_best = 0

            print()
            print(f"cheng_anchor e{epoch}")
            print(f"total loss: {val_loss:.6f} ({loss_improve})")
            print(f"bpp: {val_bpp:.6f}")
            print(f"aux loss: {val_aux:.6f}")
            print(f"psnr: {val_psnr:.3f} dB ({psnr_improve})")
            print(f"epoch time: {epoch_time:.1f} s")
            print(f"Best so far: epoch {best_epoch} val-RD {best_val_rd:.6f}")

            append_metrics_csv(
                epoch=epoch,
                train_loss=train_loss,
                train_bpp=train_bpp,
                train_aux_loss=train_aux,
                train_rd_loss=train_rd,
                train_psnr=train_psnr,
                val_loss=val_loss,
                val_bpp=val_bpp,
                val_aux_loss=val_aux,
                val_rd_loss=val_rd,
                val_psnr=val_psnr,
                val_loss_delta=val_loss_delta,
                train_val_gap=train_val_gap,
                is_best=is_best,
                epoch_time=epoch_time,
            )

        print()
        print(f"Loading RD-best weights from epoch {best_epoch}...")
        model.load_state_dict(best_state)

        # ---- Aux calibration: backbone frozen, quantile tables only. ----
        # Kept only if validation aux improves; otherwise tables revert.
        print()
        print("=" * 70)
        print("AUX CALIBRATION (backbone frozen, quantiles only)")
        print("=" * 70)

        for param_name, param in model.named_parameters():
            param.requires_grad = param_name.endswith(".quantiles")

        calib_optimizer = optim.Adam(get_aux_parameters(model), lr=AUX_LEARNING_RATE)

        baseline_quantiles = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key.endswith(".quantiles")
        }
        baseline_scores = validate_one_epoch(model, val_loader, criterion, device)
        baseline_aux = baseline_scores[4]
        baseline_rd = baseline_scores[5]

        calib_best_aux = baseline_aux
        calib_best_rd = baseline_rd
        calib_quantiles = {
            key: value.clone() for key, value in baseline_quantiles.items()
        }

        calib_steps = len(train_loader)

        for calib_epoch in range(1, CALIB_EPOCHS + 1):
            model.train()
            calib_train_sum = 0.0

            # No data forward: aux_loss() does not use image data.
            for _ in range(calib_steps):
                calib_optimizer.zero_grad(set_to_none=True)
                batch_aux = model.aux_loss()
                batch_aux.backward()
                calib_optimizer.step()
                calib_train_sum += float(batch_aux.item())

            calib_scores = validate_one_epoch(model, val_loader, criterion, device)
            calib_val_aux = calib_scores[4]
            calib_val_rd = calib_scores[5]

            # Save candidate only on aux gain that does not worsen RD,
            # so an RD-safe candidate is never overwritten by an RD-breaking one.
            if calib_val_aux < calib_best_aux and calib_val_rd <= baseline_rd:
                calib_best_aux = calib_val_aux
                calib_best_rd = calib_val_rd
                calib_quantiles = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                    if key.endswith(".quantiles")
                }

            print(
                f"AUX CALIB e{calib_epoch}/{CALIB_EPOCHS} | "
                f"train aux {calib_train_sum / calib_steps:.4f} | "
                f"val aux {calib_val_aux:.4f} | val RD {calib_val_rd:.4f}"
            )

        # Accept only if aux improved WITHOUT worsening RD (the reported metric).
        aux_calibrated = (
            calib_best_aux < baseline_aux and calib_best_rd <= baseline_rd
        )

        if aux_calibrated:
            print(
                f"Calibration improved val aux "
                f"({baseline_aux:.4f} -> {calib_best_aux:.4f}) "
                f"with val RD {baseline_rd:.4f} -> {calib_best_rd:.4f}; keeping tables."
            )
            load_quantile_tables(model, calib_quantiles)
        else:
            print("No safe aux improvement; keeping pre-calibration tables.")
            load_quantile_tables(model, baseline_quantiles)

        for param in model.parameters():
            param.requires_grad = True

        # Re-evaluate: calibration changed the tables, so stored scores may
        # no longer match the saved weights. Checkpoint gets fresh values.
        refreshed = validate_one_epoch(model, val_loader, criterion, device)
        _, fresh_bpp, _, fresh_psnr, fresh_aux, fresh_rd = refreshed
        fresh_total = fresh_rd + fresh_aux

        print(
            f"Post-calib scores: total {fresh_total:.6f} | "
            f"bpp {fresh_bpp:.6f} | psnr {fresh_psnr:.3f} dB"
        )

        (
            best_train_loss, best_train_bpp, best_train_psnr,
            best_val_loss_end, best_val_bpp, best_val_psnr,
        ) = best_scores

        save_latest_full_checkpoint(
            model=model,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            epoch=best_epoch,
            train_loss=best_train_loss,
            val_loss=fresh_total,
            val_bpp=fresh_bpp,
            val_psnr=fresh_psnr,
            aux_calibrated=aux_calibrated,
            calib_best_aux=calib_best_aux,
            calib_best_rd=calib_best_rd,
        )
        save_deploy_models(model=model, epoch=best_epoch)

        zip_and_save_models(best_epoch)

        print()
        print("Training finished.")
        print("Latest full checkpoint:", LATEST_FULL_CHECKPOINT)
        print("Final output zip:", OUTPUT_MODEL_ZIP)

    print()
    print("#" * 70)
    print("MODEL FINISHED SUCCESSFULLY.")
    print("#" * 70)


if __name__ == "__main__":
    main()
