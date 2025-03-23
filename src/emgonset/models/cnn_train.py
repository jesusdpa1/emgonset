# %% Import necessary libraries
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from emgonset.models.cnn_base import (
    DiceLoss,
    FocalLoss,
    create_emg_onset_model,
    create_improved_emg_onset_model,
)
from emgonset.processing.filters import (
    create_bandpass_filter,
    create_emg_filter,
    create_notch_filter,
)
from emgonset.processing.transforms import EMGTransformCompose
from emgonset.utils.io import (
    create_emg_dataloader,
    create_mask_dataloader,
)
from emgonset.visualization.general_plots import plot_time_series

# %% Set up data directory and preprocessing
base_dir = Path(r"E:/jpenalozaa")
data_dir = base_dir.joinpath(r"onset_detection\00_")

# Define target length for consistent data
target_length = 12000  # Adjust as needed for your dataset

# Create filters with default padding (100ms)
emg_filter = create_emg_filter(
    [
        create_notch_filter(notch_freq=60),
        create_bandpass_filter(low_cutoff=20, high_cutoff=2000),
    ]
)

# Combine them in a pipeline with fixed length transform
transform = EMGTransformCompose([emg_filter])

window_size_sec = 1.0
window_stride_sec = 0.5

# %% Create data loaders - NO SHUFFLING for time series data
emg_dataloader, fs = create_emg_dataloader(
    data_dir=data_dir,
    window_size_sec=window_size_sec,
    window_stride_sec=window_stride_sec,
    batch_size=32,
    channels=[1],
    transform=transform,
    num_workers=0,
    shuffle=False,  # No shuffling for time series
)

mask_dataloader, _ = create_mask_dataloader(
    data_dir=data_dir,
    window_size_sec=window_size_sec,
    window_stride_sec=window_stride_sec,
    batch_size=32,
    channels=[1],
    num_workers=0,
    shuffle=False,  # No shuffling for time series
)

# Calculate total batches and use only half of them
total_batches = len(emg_dataloader)
half_batches = total_batches // 2

# Split the half batches into train/val (80/20 split)
train_size = int(half_batches * 0.8)
val_size = half_batches - train_size

print(f"Data loaders created with sampling frequency: {fs} Hz")
print(f"Total batches: {total_batches}, Using half: {half_batches}")
print(f"Training: {train_size}, Validation: {val_size}")

# %% Setup model and training parameters
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters
n_fft = 256
hop_length = 64
dropout_rate = 0.3

# Create the improved model
model = create_improved_emg_onset_model(
    n_fft=n_fft, hop_length=hop_length, dropout_rate=dropout_rate
)
model = model.to(device)

# %% Define loss function and optimizer
# Combined loss function (weighted combination of Dice and Focal)
dice_loss = DiceLoss()
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Loss weights
dice_weight = 0.5
focal_weight = 0.5

# Optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)


# %% Improved training and validation functions with error handling
def train_epoch(model, emg_loader, mask_loader, optimizer, device, train_batches):
    model.train()
    total_loss = 0
    batch_count = 0
    skipped_batches = 0

    # Create iterators
    emg_iter = iter(emg_loader)
    mask_iter = iter(mask_loader)

    # Progress bar
    pbar = tqdm(total=train_batches, desc="Training")

    for batch_idx in range(train_batches):
        try:
            # Get batches with error handling
            try:
                emg_batch = next(emg_iter).to(device)
                mask_batch = next(mask_iter).to(device)

                # Check for shape consistency
                if emg_batch.shape[2] != mask_batch.shape[2]:
                    print(
                        f"Shape mismatch: EMG shape {emg_batch.shape}, Mask shape {mask_batch.shape}"
                    )
                    # Try to fix by truncating to the smaller size
                    min_length = min(emg_batch.shape[2], mask_batch.shape[2])
                    emg_batch = emg_batch[:, :, :min_length]
                    mask_batch = mask_batch[:, :, :min_length]

                # Forward pass
                outputs = model(emg_batch)

                # Ensure outputs and mask_batch have same shape
                if outputs.shape != mask_batch.shape:
                    print(
                        f"Shape mismatch: Output shape {outputs.shape}, Mask shape {mask_batch.shape}"
                    )
                    # Attempt to resize mask to match outputs
                    if outputs.shape[2] < mask_batch.shape[2]:
                        mask_batch = mask_batch[:, :, : outputs.shape[2]]
                    else:
                        outputs = outputs[:, :, : mask_batch.shape[2]]

                # Calculate loss
                d_loss = dice_loss(outputs, mask_batch)
                f_loss = focal_loss(outputs, mask_batch)
                loss = dice_weight * d_loss + focal_weight * f_loss

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()

                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Track loss
                total_loss += loss.item()
                batch_count += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "skipped": skipped_batches})

            except ValueError as e:
                print(f"ValueError in batch {batch_idx}: {e}")
                # Skip this batch
                skipped_batches += 1
                pbar.update(1)
                pbar.set_postfix({"skipped": skipped_batches})
                continue
            except RuntimeError as e:
                print(f"RuntimeError in batch {batch_idx}: {e}")
                # Skip this batch
                skipped_batches += 1
                pbar.update(1)
                pbar.set_postfix({"skipped": skipped_batches})
                continue

        except StopIteration:
            break

    pbar.close()
    print(
        f"Completed epoch with {skipped_batches} skipped batches out of {train_batches}"
    )
    return total_loss / batch_count if batch_count > 0 else 0


def validate(model, emg_loader, mask_loader, device, val_start_batch, val_batches):
    model.eval()
    total_loss = 0
    batch_count = 0
    skipped_batches = 0

    # Create iterators
    emg_iter = iter(emg_loader)
    mask_iter = iter(mask_loader)

    # Skip to validation batch start
    for _ in range(val_start_batch):
        try:
            next(emg_iter)
            next(mask_iter)
        except StopIteration:
            print("Not enough data to skip to validation batches")
            return 0

    # Progress bar
    pbar = tqdm(total=val_batches, desc="Validation")

    with torch.no_grad():
        for _ in range(val_batches):
            try:
                # Get batches with error handling
                try:
                    emg_batch = next(emg_iter).to(device)
                    mask_batch = next(mask_iter).to(device)

                    # Check for shape consistency
                    if emg_batch.shape[2] != mask_batch.shape[2]:
                        print(
                            f"Shape mismatch: EMG shape {emg_batch.shape}, Mask shape {mask_batch.shape}"
                        )
                        # Try to fix by truncating to the smaller size
                        min_length = min(emg_batch.shape[2], mask_batch.shape[2])
                        emg_batch = emg_batch[:, :, :min_length]
                        mask_batch = mask_batch[:, :, :min_length]

                    # Forward pass
                    outputs = model(emg_batch)

                    # Ensure outputs and mask_batch have same shape
                    if outputs.shape != mask_batch.shape:
                        print(
                            f"Shape mismatch: Output shape {outputs.shape}, Mask shape {mask_batch.shape}"
                        )
                        # Attempt to resize mask to match outputs
                        if outputs.shape[2] < mask_batch.shape[2]:
                            mask_batch = mask_batch[:, :, : outputs.shape[2]]
                        else:
                            outputs = outputs[:, :, : mask_batch.shape[2]]

                    # Calculate loss
                    d_loss = dice_loss(outputs, mask_batch)
                    f_loss = focal_loss(outputs, mask_batch)
                    loss = dice_weight * d_loss + focal_weight * f_loss

                    # Track loss
                    total_loss += loss.item()
                    batch_count += 1

                    # Update progress bar
                    pbar.update(1)

                except ValueError as e:
                    print(f"ValueError in validation batch: {e}")
                    skipped_batches += 1
                    pbar.update(1)
                    continue
                except RuntimeError as e:
                    print(f"RuntimeError in validation batch: {e}")
                    skipped_batches += 1
                    pbar.update(1)
                    continue

            except StopIteration:
                break

    pbar.close()
    print(
        f"Completed validation with {skipped_batches} skipped batches out of {val_batches}"
    )
    return total_loss / batch_count if batch_count > 0 else 0


# %% Training loop
# Training parameters
num_epochs = 1
best_val_loss = float("inf")
patience = 10  # Early stopping patience
patience_counter = 0

# Lists to store metrics
train_losses = []
val_losses = []
epoch_times = []

print("Starting training...")
for epoch in range(num_epochs):
    # Time each epoch
    start_time = time.time()

    # Reset iterators for each epoch
    emg_dataloader_reset, _ = create_emg_dataloader(
        data_dir=data_dir,
        window_size_sec=window_size_sec,
        window_stride_sec=window_stride_sec,
        batch_size=32,
        channels=[1],
        transform=transform,
        num_workers=0,
        shuffle=False,
    )

    mask_dataloader_reset, _ = create_mask_dataloader(
        data_dir=data_dir,
        window_size_sec=window_size_sec,
        window_stride_sec=window_stride_sec,
        batch_size=32,
        channels=[1],
        num_workers=0,
        shuffle=False,
    )

    # Train on the first part of the half of batches
    train_loss = train_epoch(
        model,
        emg_dataloader_reset,
        mask_dataloader_reset,
        optimizer,
        device,
        train_size,
    )
    train_losses.append(train_loss)

    # Validate on the remaining part of the half of batches
    val_loss = validate(
        model,
        emg_dataloader_reset,
        mask_dataloader_reset,
        device,
        train_size,  # Skip training batches
        val_size,
    )
    val_losses.append(val_loss)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    # End timing
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Time: {epoch_time:.2f}s, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the best model
        torch.save(model.state_dict(), "best_emg_onset_model.pt")
        patience_counter = 0
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# %% Plot training and validation loss
plt.figure(figsize=(16, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Plot epoch times
plt.subplot(1, 2, 2)
plt.plot(epoch_times, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.title("Epoch Training Time")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300)
plt.show()

# %% Load the best model and make predictions on validation data
# Load the best model
model.load_state_dict(torch.load("best_emg_onset_model.pt"))
# %%
model.eval()
# %%
# Create fresh data loaders
emg_dataloader_eval, _ = create_emg_dataloader(
    data_dir=data_dir,
    window_size_sec=window_size_sec,
    window_stride_sec=window_stride_sec,
    batch_size=32,
    channels=[1],
    transform=transform,
    num_workers=0,
    shuffle=False,
)

mask_dataloader_eval, _ = create_mask_dataloader(
    data_dir=data_dir,
    window_size_sec=window_size_sec,
    window_stride_sec=window_stride_sec,
    batch_size=32,
    channels=[1],
    num_workers=0,
    shuffle=False,
)

# Get iterators
emg_iter = iter(emg_dataloader_eval)
mask_iter = iter(mask_dataloader_eval)

# Get a validation batch
emg_batch = next(emg_iter).to(device)
mask_batch = next(mask_iter).to(device)

# Get predictions
with torch.no_grad():
    predictions = model(emg_batch)

# Make sure predictions and mask have the same shape
if predictions.shape[2] != mask_batch.shape[2]:
    min_length = min(predictions.shape[2], mask_batch.shape[2])
    predictions = predictions[:, :, :min_length]
    mask_batch = mask_batch[:, :, :min_length]

# Convert to numpy for plotting
emg_sample = emg_batch[0].cpu().numpy()
mask_sample = mask_batch[0].cpu().numpy()
prediction_sample = predictions[0].cpu().numpy()

# %% Plot the results
plt.figure(figsize=(15, 10))

# Plot EMG signal
plt.subplot(3, 1, 1)
plt.plot(emg_sample[0])
plt.title("EMG Signal")
plt.grid(True)

# Plot ground truth mask
plt.subplot(3, 1, 2)
plt.plot(mask_sample[0])
plt.title("Ground Truth Mask")
plt.grid(True)

# Plot predicted mask
plt.subplot(3, 1, 3)
plt.plot(prediction_sample[0])
plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)  # Threshold line
plt.title("Predicted Mask")
plt.grid(True)

plt.tight_layout()
plt.savefig("prediction_results.png", dpi=300)
plt.show()


# %% Calculate and display metrics for validation data
def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate performance metrics for binary classification"""
    # Apply threshold to predictions
    binary_preds = (predictions > threshold).float()

    # Calculate metrics
    true_positives = (binary_preds * targets).sum().item()
    false_positives = (binary_preds * (1 - targets)).sum().item()
    false_negatives = ((1 - binary_preds) * targets).sum().item()
    true_negatives = ((1 - binary_preds) * (1 - targets)).sum().item()

    # Precision, recall, and F1
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Accuracy
    total = true_positives + false_positives + false_negatives + true_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


# Evaluate model on validation set (second part of the first half of data)
def evaluate_validation_set(model, dataloader_maker, batch_size=32, skip_batches=0):
    """Evaluate model on the validation set with detailed metrics"""
    model.eval()

    # Create dataloaders
    emg_loader, _ = dataloader_maker[0](
        data_dir=data_dir,
        window_size_sec=window_size_sec,
        window_stride_sec=window_stride_sec,
        batch_size=batch_size,
        channels=[1],
        transform=transform,
        num_workers=0,
        shuffle=False,
    )

    mask_loader, _ = dataloader_maker[1](
        data_dir=data_dir,
        window_size_sec=window_size_sec,
        window_stride_sec=window_stride_sec,
        batch_size=batch_size,
        channels=[1],
        num_workers=0,
        shuffle=False,
    )

    # Create iterators
    emg_iter = iter(emg_loader)
    mask_iter = iter(mask_loader)

    # Skip batches to reach validation set
    for _ in range(skip_batches):
        try:
            next(emg_iter)
            next(mask_iter)
        except StopIteration:
            print("Warning: Couldn't skip batches. Using from the beginning.")
            emg_iter = iter(emg_loader)
            mask_iter = iter(mask_loader)
            break

    # Collect all predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for _ in tqdm(range(val_size), desc="Evaluating"):
            try:
                emg_batch = next(emg_iter).to(device)
                mask_batch = next(mask_iter).to(device)

                # Get predictions
                predictions = model(emg_batch)

                # Handle any shape inconsistencies
                min_length = min(predictions.shape[2], mask_batch.shape[2])
                predictions = predictions[:, :, :min_length]
                mask_batch = mask_batch[:, :, :min_length]

                # Store results
                all_preds.append(predictions.cpu())
                all_targets.append(mask_batch.cpu())

            except (StopIteration, ValueError, RuntimeError) as e:
                print(f"Error during evaluation: {e}")
                continue

    # Concatenate results
    if all_preds:
        # Get first batch shape to handle concatenation properly
        first_pred_shape = all_preds[0].shape

        # Concatenate along batch dimension if shapes are consistent
        if all(p.shape[1:] == first_pred_shape[1:] for p in all_preds):
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Calculate metrics
            metrics = calculate_metrics(all_preds, all_targets)

            # Calculate metrics at different thresholds
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            threshold_metrics = {
                t: calculate_metrics(all_preds, all_targets, threshold=t)
                for t in thresholds
            }

            return metrics, threshold_metrics
        else:
            print("Warning: Inconsistent shapes found. Evaluation may be incomplete.")
            return None, None
    else:
        print("No valid batches for evaluation.")
        return None, None


# Run evaluation on the validation set (second part of the half of data)
print("Evaluating model on validation data...")
metrics, threshold_metrics = evaluate_validation_set(
    model,
    (create_emg_dataloader, create_mask_dataloader),
    batch_size=32,
    skip_batches=train_size,  # Skip the training batches
)

# Print metrics
if metrics:
    print("\nValidation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Negatives: {metrics['true_negatives']}")

    # Print metrics at different thresholds
    print("\nMetrics at different thresholds:")
    for threshold, m in threshold_metrics.items():
        print(f"\nThreshold: {threshold}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall: {m['recall']:.4f}")
        print(f"F1 Score: {m['f1_score']:.4f}")
        print(f"Accuracy: {m['accuracy']:.4f}")
