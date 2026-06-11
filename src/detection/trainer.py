# -*- coding: utf-8 -*-
"""
Training and evaluation engine

A single fit routine drives both the full model and the ablation conditions: it
runs the epoch loop, tracks the best validation macro-F1, and restores the best
weights. Evaluation returns predictions, softmax probabilities and original split
indices so metrics and the qualitative join can both be derived from one pass
"""

import os

import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()     # Silence the per-load report and benign 513>512 tokenizer warning

from detection.config import (
    DEVICE, MODEL_NAME, NUM_LABELS, BATCH_SIZE, GRAD_ACCUM,
    LR, WARMUP_FRAC, WEIGHT_DECAY,
)
from detection.datasets import PubHealthDataset, AblationDataset
from detection.inputs import ABLATION_VARIANTS
from detection.losses import make_criterion
from detection.weights import compute_class_weights
from detection.sampling import oversample_minority
from detection.metrics import macro_f1, report_dict
from detection.seeding import set_seed, SEED
from detection.paths import experiment_model_dir

# BF16 autocast: the L40S has native bfloat16 tensor cores, which roughly halve
# activation memory and speed up the forward/backward without the loss scaling that
# fp16 would require. Enabled on CUDA only; the context manager is a no-op on CPU
AMP_DTYPE = torch.bfloat16
USE_AMP = (DEVICE.type == "cuda")


def _build_model():
    """
    Instantiate a fresh RoBERTa sequence classifier on the active device
    """
    return RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)


def _criterion_for(cfg, train_records):
    """
    Build the criterion, deriving class weights from the original training split
    """
    weights = None
    if cfg.use_class_weights:
        weights = compute_class_weights(train_records, temperature=cfg.weight_temperature)
    return make_criterion(cfg, weights)


def _train_one_epoch(model, loader, criterion, optimizer, scheduler) -> float:
    """
    Run one training epoch with gradient accumulation and return the mean loss
    """
    model.train()
    optimizer.zero_grad()
    running = 0.0
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels) / GRAD_ACCUM
        loss.backward()
        running += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return running / len(loader)


def _predict(model, loader, criterion=None) -> dict:
    """
    Run inference over a loader and collect labels, predictions, probabilities

    Args:
        model: The classifier to evaluate
        loader (DataLoader): Batches carrying input_ids, attention_mask, label, orig_idx
        criterion: Optional loss used to also report a mean validation loss
    Returns:
        dict: y_true, y_pred, probs, indices and an optional mean loss
    """
    model.eval()
    y_true, y_pred, probs_all, indices = [], [], [], []
    loss_total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits.float()                 # Back to fp32 for stable loss/softmax and numpy
            if criterion is not None:
                loss_total += criterion(logits, labels).item()
                n_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            y_pred.extend(probs.argmax(axis=-1).tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            probs_all.extend(probs.tolist())
            indices.extend(batch["orig_idx"].numpy().tolist())

    avg_loss = (loss_total / n_batches) if n_batches else None
    return {"y_true": y_true, "y_pred": y_pred, "probs": probs_all,
            "indices": indices, "loss": avg_loss}


def _fit(model, train_loader, val_loader, criterion, epochs, ckpt_path=None) -> dict:
    """
    Train a model, tracking and restoring the best-validation-F1 weights

    When ckpt_path is given the best weights are persisted to disk and reloaded;
    otherwise they are held in memory, which suits the short ablation runs

    Args:
        model: The classifier to train in place
        train_loader (DataLoader): Training batches
        val_loader (DataLoader): Validation batches
        criterion: Loss module
        epochs (int): Number of epochs to train
        ckpt_path (str): Optional path to save the best checkpoint
    Returns:
        dict: history, best_val_f1 and best_epoch
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRAC), total_steps
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1, best_epoch, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        out = _predict(model, val_loader, criterion)
        val_f1 = macro_f1(out["y_true"], out["y_pred"])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(out["loss"])
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={out['loss']:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  new best (val_f1={best_val_f1:.4f})")

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    elif best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_f1": best_val_f1, "best_epoch": best_epoch}


def _make_loader(dataset, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader with deterministic worker seeding
    """
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2,
        worker_init_fn=lambda wid: set_seed(SEED + wid),
    )


def train_full(cfg, train_records, val_records) -> dict:
    """
    Train the full-evidence model for an experiment and checkpoint the best weights

    Args:
        cfg (ExperimentConfig): Experiment specification
        train_records (list): Original training split
        val_records (list): Validation split
    Returns:
        dict: The trained model under key model plus history, best_val_f1,
            best_epoch and the checkpoint path
    """
    set_seed(SEED)
    fit_records = train_records
    if cfg.oversample_factors:
        print("Oversampling minority classes for the full run")
        fit_records = oversample_minority(train_records, cfg.oversample_factors)

    print("Building datasets (evidence retrieval happens here)")
    train_loader = _make_loader(PubHealthDataset(fit_records, cfg), shuffle=True)
    val_loader = _make_loader(PubHealthDataset(val_records, cfg), shuffle=False)

    criterion = _criterion_for(cfg, train_records)        # Weights from the original split
    model = _build_model()
    ckpt = os.path.join(experiment_model_dir(cfg.name), "best_model.pt")

    summary = _fit(model, train_loader, val_loader, criterion, cfg.epochs, ckpt_path=ckpt)
    summary["model"] = model
    summary["checkpoint"] = ckpt
    print(f"Best validation macro-F1: {summary['best_val_f1']:.4f}")
    return summary


def evaluate_full(cfg, records, model=None, checkpoint=None) -> dict:
    """
    Evaluate the full-evidence model on a split and return raw predictions

    Args:
        cfg (ExperimentConfig): Experiment specification
        records (list): Split to evaluate, typically the test split
        model: An in-memory model, used in preference to checkpoint when given
        checkpoint (str): Path to load weights from when model is None
    Returns:
        dict: y_true, y_pred, probs and indices for the split
    """
    if model is None:
        model = _build_model()
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    loader = _make_loader(PubHealthDataset(records, cfg), shuffle=False)
    return _predict(model, loader)


def run_ablation(cfg, train_records, val_records, test_records) -> dict:
    """
    Train and evaluate the three input variants for an experiment

    Each variant is trained from scratch for the ablation epoch budget and scored
    on the test split with the best-validation weights

    Args:
        cfg (ExperimentConfig): Experiment specification
        train_records (list): Original training split
        val_records (list): Validation split
        test_records (list): Test split
    Returns:
        dict: Per-variant val_best, test_macro_f1, per-class metrics and confusion
    """
    results = {}
    for variant in ABLATION_VARIANTS:
        print(f"\nAblation variant: {variant}")
        set_seed(SEED)
        fit_records = train_records
        if cfg.oversample_factors:
            fit_records = oversample_minority(train_records, cfg.oversample_factors)

        train_loader = _make_loader(AblationDataset(fit_records, variant, cfg), shuffle=True)
        val_loader = _make_loader(AblationDataset(val_records, variant, cfg), shuffle=False)
        test_loader = _make_loader(AblationDataset(test_records, variant, cfg), shuffle=False)

        criterion = _criterion_for(cfg, train_records)
        model = _build_model()
        summary = _fit(model, train_loader, val_loader, criterion, cfg.epochs_ablation)

        test_out = _predict(model, test_loader)
        report = report_dict(test_out["y_true"], test_out["y_pred"])
        results[variant] = {
            "val_best": summary["best_val_f1"],
            "test_macro_f1": report["macro_f1"],
            "accuracy": report["accuracy"],
            "per_class": report["per_class"],
            "confusion_matrix": report["confusion_matrix"],
        }
        print(f"  {variant} test macro-F1 = {report['macro_f1']:.4f}")
    return results
