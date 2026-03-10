# Standard library
import os
import sys
import time
import yaml
import argparse

# --- ARGS FIRST ---
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='experiments/configs/unet_luserna.yaml')
parser.add_argument('--rfinln_path', type=str, default='../RFI-NLN')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--truth_path', type=str, default=None)
args_cli = parser.parse_args()

sys.path.insert(0, args_cli.rfinln_path)
sys.path.insert(0, 'src')

# Third party
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# RFI-NLN
from utils.data import patches, get_patches, get_lofar_data
from utils.data.processor import process

# Project
from loaders.luserna_loader import load_luserna, load_luserna_truth
from loaders.lofar_loader import Args
from preprocessing.spectrogram import polynomial_detrend, extract_and_split_patches, balance_dataset
from methods.dl.unet import build_unet, train_unet
from evaluation.timing import Timer
from visualization.plots import plot_detection_result, plot_rfi_distribution

# --- CONFIG ---
with open(args_cli.config) as f:
    cfg = yaml.safe_load(f)

if args_cli.data_path is not None:
    cfg['data_path'] = args_cli.data_path
if args_cli.truth_path is not None:
    cfg['truth_path'] = args_cli.truth_path

METHOD = cfg['method']
DATASET = cfg['dataset']
RANDOM_SEED = cfg['random_seed'] if 'random_seed' in cfg else 42

# --- LOAD & PREPROCESSING ---
if DATASET == 'luserna':
    print('Loading Luserna data...')
    data = load_luserna(cfg['name'], path=cfg['data_path'], powers=False)
    truth = load_luserna_truth(path=cfg['truth_path'])

    data_np = data.to_numpy().astype(np.float32)
    truth_np = truth.to_numpy().astype(bool)

    print('Preprocessing...')
    data_np = polynomial_detrend(data_np, degree=2)
    data_np = np.clip(data_np, -10, 10)
    data_np = process(data_np[..., np.newaxis], per_image=False)[..., 0]

    train_data, train_masks, test_data, test_masks = extract_and_split_patches(
        data_np, truth_np,
        patch_size=tuple(cfg['patch_size_512']),
        train_size=cfg['train_size'],
        max_patches=cfg['max_patches'],
        random_seed=RANDOM_SEED
    )

    train_data, train_masks = balance_dataset(train_data, train_masks, random_seed=RANDOM_SEED)

    # Second patching 32x32
    p_size = (1, *cfg['patch_size'], 1)
    s_size = (1, *cfg['patch_size'], 1)
    rate = (1, 1, 1, 1)

    train_data = get_patches(train_data, None, p_size, s_size, rate, 'VALID')
    train_masks = get_patches(train_masks.astype('float32'), None, p_size, s_size, rate, 'VALID')
    test_data_p = get_patches(test_data, None, p_size, s_size, rate, 'VALID')
    test_masks_p = get_patches(test_masks.astype('float32'), None, p_size, s_size, rate, 'VALID')

elif DATASET == 'lofar':
    print('Loading LOFAR data...')
    args = Args()
    args.data_path = cfg['data_path']
    args.update_input_shape()

    train_data, train_masks, test_data_p, test_masks_p = get_lofar_data(args)

    train_data = train_data[:cfg['max_samples'], ...]
    train_masks = train_masks[:cfg['max_samples'], ...]
    test_data_p = test_data_p[:cfg['max_samples'], ...]
    test_masks_p = test_masks_p[:cfg['max_samples'], ...]

    print('Preprocessing...')
    train_data = np.clip(train_data, cfg['clip_min'], cfg['clip_max'])
    train_data = process(train_data, per_image=False)
    test_data_p = np.clip(test_data_p, cfg['clip_min'], cfg['clip_max'])
    test_data_p = process(test_data_p, per_image=False)

    test_data = test_data_p

    # Patching 32x32
    p_size = (1, *cfg['patch_size'], 1)
    s_size = (1, *cfg['patch_size'], 1)
    rate = (1, 1, 1, 1)

    train_data = get_patches(train_data, None, p_size, s_size, rate, 'VALID')
    train_masks = get_patches(train_masks.astype('float32'), None, p_size, s_size, rate, 'VALID')
    test_data_p = get_patches(test_data_p, None, p_size, s_size, rate, 'VALID')
    test_masks_p = get_patches(test_masks_p.astype('float32'), None, p_size, s_size, rate, 'VALID')

else:
    raise ValueError(f'Unknown dataset: {DATASET}')

plot_rfi_distribution(train_data, train_masks, title=f'{DATASET.upper()} - RFI vs Clean')
print(f'Train data shape: {train_data.shape}')
print(f'Test data shape: {test_data_p.shape}')

# --- BUILD ARGS FOR MODEL ---
args = Args()
args.patch_x = cfg['patch_size'][0] if isinstance(cfg['patch_size'], list) else cfg['patch_size']
args.patch_y = cfg['patch_size'][1] if isinstance(cfg['patch_size'], list) else cfg['patch_size']
args.update_input_shape()

# --- BUILD MODEL ---
model = build_unet(
    args=args,
    n_filters=cfg['n_filters'],
    dropout=cfg['dropout'],
    batchnorm=cfg['batchnorm']
)

# --- TRAIN ---
print(f'Training {METHOD.upper()} for {cfg["epochs"]} epochs...')
with Timer() as t_train:
    model = train_unet(
        model=model,
        train_data=train_data,
        train_masks=train_masks,
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        learning_rate=cfg['learning_rate'],
        buffer_size=cfg['buffer_size']
    )

print(f'Training time: {t_train}')

# --- SAVE MODEL ---
os.makedirs('models', exist_ok=True)
model.save(cfg['models_path'])
print(f'Model saved to {cfg["models_path"]}')

# --- INFERENCE ---
print('Running inference...')
with Timer() as t_inf:
    pred_patches = model.predict(test_data_p)

pred_recon = patches.reconstruct(pred_patches, args)
test_masks_recon = patches.reconstruct(test_masks_p, args)

# --- METRICS ---
y_true = test_masks_recon.reshape(-1)
y_pred_continuous = pred_recon.reshape(-1)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_continuous)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_continuous)

f1_scores = [
    2 * (p * r) / (p + r) if (p + r) > 0 else 0
    for p, r in zip(precision_curve[:-1], recall_curve[:-1])
]
optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx]

auroc = auc(fpr, tpr)
auprc = auc(recall_curve, precision_curve)
best_precision = precision_curve[optimal_idx]
best_recall = recall_curve[optimal_idx]
best_f1 = f1_scores[optimal_idx]

print(f'\nResults:')
print(f'  Precision : {best_precision:.4f}')
print(f'  Recall    : {best_recall:.4f}')
print(f'  F1        : {best_f1:.4f}')
print(f'  AUROC     : {auroc:.4f}')
print(f'  AUPRC     : {auprc:.4f}')
print(f'  Threshold : {optimal_threshold:.4f}')
print(f'  Train time: {t_train.elapsed:.2f}s')

# --- SAVE RESULTS ---
os.makedirs(os.path.dirname(cfg['results_path']), exist_ok=True)
metrics = {
    'method': METHOD,
    'dataset': DATASET,
    'epochs': cfg['epochs'],
    'n_filters': cfg['n_filters'],
    'learning_rate': cfg['learning_rate'],
    'dropout': cfg['dropout'],
    'batch_size': cfg['batch_size'],
    'auroc': auroc,
    'auprc': auprc,
    'threshold': optimal_threshold,
    'precision': best_precision,
    'recall': best_recall,
    'f1': best_f1,
    'train_time_seconds': t_train.elapsed,
    'inference_time_seconds': t_inf.elapsed
}
pd.DataFrame([metrics]).to_csv(cfg['results_path'], index=False)
print(f'Results saved to {cfg["results_path"]}')

# --- PLOT ---
sample_idx = 0
pred_mask_sample = (pred_recon[sample_idx] > optimal_threshold).astype(bool)

plot_detection_result(
    spectrogram=test_data[sample_idx],
    ground_truth=test_masks_recon[sample_idx].astype(bool),
    predicted_mask=pred_mask_sample,
    title=f'{METHOD.upper()} {DATASET.upper()} — threshold={optimal_threshold:.2f}',
    save_path=cfg['plot_path']
)