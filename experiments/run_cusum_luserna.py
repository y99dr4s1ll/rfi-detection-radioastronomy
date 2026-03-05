import numpy as np
import pandas as pd
import yaml
import argparse
from data.luserna_loader import load_luserna, load_luserna_truth
from preprocessing.spectrogram import polynomial_detrend, extract_and_split_patches
from methods.statistical.cusum import CUSUM
from evaluation.metrics import compute_metrics
from evaluation.timing import Timer
from visualization.plots import plot_detection_result

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='experiments/configs/cusum_luserna.yaml')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

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
import os
os.makedirs(os.path.dirname(cfg['results_path']), exist_ok=True)
pd.DataFrame([metrics]).to_csv(cfg['results_path'], index=False)
print(f'\nResults saved to {cfg["results_path"]}')

# --- PLOT ---
rfi_patches = np.where(np.any(test_masks > 0, axis=(1, 2, 3)))[0]
sample_idx = rfi_patches[0] if len(rfi_patches) > 0 else 0
plot_detection_result(
    spectrogram=test_data[sample_idx],
    ground_truth=test_masks[sample_idx],
    predicted_mask=pred_masks[sample_idx],
    title=f'CUSUM — k={K}, h={H}',
    save_path=cfg['plot_path']
)