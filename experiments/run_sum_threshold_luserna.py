# Standard library
import os
import sys
import yaml
import argparse

# Third party
import numpy as np
import pandas as pd

# Project
sys.path.insert(0, 'src')
from loaders.luserna_loader import load_luserna, load_luserna_truth
from preprocessing.spectrogram import polynomial_detrend, extract_and_split_patches
from methods.statistical.sum_threshold import (
    winsorized_mode,
    calculate_thresholds,
    sumthreshold_optimized
)
from evaluation.metrics import compute_metrics
from evaluation.timing import Timer
from visualization.plots import plot_detection_result, plot_rfi_distribution

# --- CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='experiments/configs/sum_threshold_luserna.yaml')
parser.add_argument('--iterations', type=int, default=None)
parser.add_argument('--molt', type=float, default=None)
args_cli = parser.parse_args()

with open(args_cli.config) as f:
    cfg = yaml.safe_load(f)

if args_cli.iterations is not None:
    cfg['iterations'] = args_cli.iterations
if args_cli.molt is not None:
    cfg['molt'] = args_cli.molt

# --- LOAD ---
print('Loading data...')
data = load_luserna(cfg['name'], path=cfg['data_path'], powers=False)
truth = load_luserna_truth(path=cfg['truth_path'])

data_np = data.to_numpy().astype(np.float32)
truth_np = truth.to_numpy().astype(bool)

# --- PREPROCESSING ---
print('Preprocessing...')
data_np = polynomial_detrend(data_np, degree=2)
data_np = np.clip(data_np, -10, 10)

_, _, test_data, test_masks = extract_and_split_patches(
    data_np, truth_np,
    patch_size=tuple(cfg['patch_size']),
    train_size=cfg['train_size'],
    max_patches=cfg['max_patches'],
    random_seed=cfg['random_seed']
)

plot_rfi_distribution(test_data, test_masks, title='Luserna - RFI vs Clean distribution')

# --- INFERENCE ---
ITERATIONS = cfg['iterations']
MOLT = cfg['molt']
WINSORIZE_LIMITS = cfg['winsorize_limits']

print(f'Running SumThreshold (iterations={ITERATIONS}, molt={MOLT}) on {test_data.shape[0]} patches...')

pred_masks = np.zeros(test_masks.shape, dtype=bool)

with Timer() as t:
    for v in range(test_data.shape[0]):
        input_arr = test_data[v, ..., 0]
        mode = winsorized_mode(input_arr, limits=WINSORIZE_LIMITS) * MOLT
        chi = calculate_thresholds(iterations=ITERATIONS, chi0=mode)
        pred_masks[v, ..., 0] = sumthreshold_optimized(input_arr, chi)

print(f'Inference time: {t}')

# --- METRICS ---
metrics = compute_metrics(test_masks, pred_masks)
metrics['time_seconds'] = t.elapsed
metrics['iterations'] = ITERATIONS
metrics['molt'] = MOLT
metrics['dataset'] = cfg['dataset']
metrics['method'] = 'sum_threshold'

print(f"\nResults:")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1        : {metrics['f1']:.4f}")
print(f"  Time      : {metrics['time_seconds']:.2f}s")

# --- SAVE ---
os.makedirs(os.path.dirname(cfg['results_path']), exist_ok=True)
pd.DataFrame([metrics]).to_csv(cfg['results_path'], index=False)
print(f'\nResults saved to {cfg["results_path"]}')

# --- PLOT ---
rfi_patches = np.where(np.any(test_masks > 0, axis=(1, 2, 3)))[0]
sample_idx = rfi_patches[0] if len(rfi_patches) > 0 else 0
print(f'Plotting patch {sample_idx} (RFI patches available: {len(rfi_patches)})')

plot_detection_result(
    spectrogram=test_data[sample_idx],
    ground_truth=test_masks[sample_idx],
    predicted_mask=pred_masks[sample_idx],
    title=f'SumThreshold — iterations={ITERATIONS}, molt={MOLT}',
    save_path=cfg['plot_path']
)