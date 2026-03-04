# Standard library
import os
import sys
import yaml
import argparse

# Third party
import numpy as np
import pandas as pd

# RFI-NLN
sys.path.insert(0, '/kaggle/working/RFI-NLN')
from utils.data import get_lofar_data

# Project
sys.path.insert(0, 'src')
from loaders.lofar_loader import Args
from methods.statistical.cusum import CUSUM
from evaluation.metrics import compute_metrics
from evaluation.timing import Timer
from visualization.plots import plot_detection_result, plot_rfi_distribution

# --- CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='experiments/configs/cusum_lofar.yaml')
args_cli = parser.parse_args()

with open(args_cli.config) as f:
    cfg = yaml.safe_load(f)

# --- LOAD ---
print('Loading LOFAR data...')
args = Args()
args.data_path = cfg['data_path']
args.update_input_shape()

train_data, train_masks, test_data, test_masks = get_lofar_data(args)

test_data = test_data[:cfg['max_samples'], ...]
test_masks = test_masks[:cfg['max_samples'], ...]
print(f'Test data shape: {test_data.shape}')

# --- PREPROCESSING ---
print('Preprocessing...')
from utils.data.processor import process

test_data = np.clip(test_data, cfg['clip_min'], cfg['clip_max'])
test_data = process(test_data, per_image=False)

plot_rfi_distribution(test_data, test_masks, title='LOFAR - RFI vs Clean distribution')

# --- INFERENCE ---
K = cfg['k']
H = cfg['h']
print(f'Running CUSUM (k={K}, h={H}) on {test_data.shape[0]} patches...')

pred_masks = np.zeros(test_masks.shape, dtype=bool)

with Timer() as t:
    for v in range(test_data.shape[0]):
        pred_masks[v, ..., 0] = CUSUM(test_data[v, ..., 0], k=K, h=H, output=False)

print(f'Inference time: {t}')

# --- METRICS ---
metrics = compute_metrics(test_masks, pred_masks)
metrics['time_seconds'] = t.elapsed
metrics['k'] = K
metrics['h'] = H
metrics['dataset'] = cfg['dataset']
metrics['method'] = 'cusum'

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
    title=f'CUSUM LOFAR — k={K}, h={H}',
    save_path=cfg['plot_path']
)