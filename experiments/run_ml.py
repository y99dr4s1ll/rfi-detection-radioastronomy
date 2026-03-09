# Standard library
import os
import sys
import yaml
import argparse
import time

# Third party
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# RFI-NLN
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='experiments/configs/knn_luserna.yaml')
parser.add_argument('--rfinln_path', type=str, default='../RFI-NLN')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--skip_features', action='store_true',
                    help='Skip feature extraction and load from CSV')
args_cli = parser.parse_args()

sys.path.insert(0, args_cli.rfinln_path)
sys.path.insert(0, 'src')

from utils.data import get_patches
from utils.data.processor import process
from loaders.luserna_loader import load_luserna, load_luserna_truth
from preprocessing.spectrogram import polynomial_detrend, extract_and_split_patches, balance_dataset
from methods.ml.features import prepare_features, create_features_dataframe, reconstruct_from_patches
from evaluation.metrics import compute_metrics, custom_f1_score_ml
from evaluation.timing import Timer
from visualization.plots import plot_detection_result, plot_rfi_distribution

# --- CONFIG ---
with open(args_cli.config) as f:
    cfg = yaml.safe_load(f)

if args_cli.data_path is not None:
    cfg['data_path'] = args_cli.data_path

METHOD = cfg['method']
PATCH_SIZE = cfg['patch_size']
IMG_SIZE = cfg['img_size']
BATCH_SIZE = cfg['batch_size']
RANDOM_SEED = cfg['random_seed']

# --- LOAD ---
print('Loading data...')
data = load_luserna(cfg['name'], path=cfg['data_path'], powers=False)
truth = load_luserna_truth(path=cfg['truth_path'])

data_np = data.to_numpy().astype(np.float32)
truth_np = truth.to_numpy().astype(bool)

# --- PREPROCESSING 512x512 ---
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

plot_rfi_distribution(train_data, train_masks, title=f'{cfg["dataset"].upper()} - RFI vs Clean')

# --- BALANCE 512x512 ---
train_data_red, train_masks_red = balance_dataset(train_data, train_masks, random_seed=RANDOM_SEED)
train_data_or = train_data.copy()
train_masks_or = train_masks.copy()

# --- PATCHES 8x8 ---
print(f'Extracting {PATCH_SIZE}x{PATCH_SIZE} patches...')
p_size = (1, PATCH_SIZE, PATCH_SIZE, 1)
s_size = (1, PATCH_SIZE, PATCH_SIZE, 1)
rate = (1, 1, 1, 1)

train_data_p = get_patches(train_data_red, None, p_size, s_size, rate, 'VALID')
train_masks_p = get_patches(train_masks_red.astype('float32'), None, p_size, s_size, rate, 'VALID').astype(bool)
test_data_p = get_patches(test_data, None, p_size, s_size, rate, 'VALID')
test_masks_p = get_patches(test_masks.astype('float32'), None, p_size, s_size, rate, 'VALID').astype(bool)

# --- FEATURE EXTRACTION ---
os.makedirs(os.path.dirname(cfg['features_train_path']), exist_ok=True)

if args_cli.skip_features and os.path.exists(cfg['features_train_path']):
    print('Loading features from cache...')
    df_train = pd.read_csv(cfg['features_train_path'])
    df_test = pd.read_csv(cfg['features_test_path'])
    X = df_train.drop(columns=['label']).values
    y = df_train['label'].values
    X_test = df_test.drop(columns=['label']).values
    y_test = df_test['label'].values
else:
    print('Extracting features...')
    X, y = prepare_features(train_data_p, train_masks_p, k_cusum=cfg['k_cusum'])
    X_test, y_test = prepare_features(test_data_p, test_masks_p, k_cusum=cfg['k_cusum'])
    create_features_dataframe(X, y).to_csv(cfg['features_train_path'], index=False)
    create_features_dataframe(X_test, y_test).to_csv(cfg['features_test_path'], index=False)
    print(f'Features saved to {cfg["features_train_path"]}')

# --- UNDERSAMPLING ---
n_positive = int(np.sum(y == 1))
n_negative = n_positive * cfg['undersampling_ratio']
undersampler = RandomUnderSampler(
    sampling_strategy={0: min(n_negative, int(np.sum(y == 0))), 1: n_positive},
    random_state=RANDOM_SEED
)
X_balanced, y_balanced = undersampler.fit_resample(X, y)
print(f'Balanced dataset: {X_balanced.shape[0]} samples')

# --- BUILD AND TRAIN MODEL ---
if METHOD == 'knn':
    from methods.ml.knn import build_knn, train_knn
    model = build_knn(
        n_neighbors=cfg['n_neighbors'],
        weights=cfg['weights'],
        p=cfg['p'],
        leaf_size=cfg['leaf_size'],
        algorithm=cfg['algorithm']
    )
elif METHOD == 'rf':
    from methods.ml.random_forest import build_rf, train_rf
    model = build_rf(
        n_estimators=cfg['n_estimators'],
        max_depth=cfg.get('max_depth', None),
        min_samples_split=cfg.get('min_samples_split', 2),
        min_samples_leaf=cfg.get('min_samples_leaf', 1),
        max_features=cfg.get('max_features', None)
    )
else:
    raise ValueError(f'Unknown method: {METHOD}')

with Timer() as t_train:
    if METHOD == 'knn':
        model = train_knn(model, X_balanced, y_balanced)
    else:
        model = train_rf(model, X_balanced, y_balanced)

print(f'Training time: {t_train}')

# --- INFERENCE ---
print('Running inference...')
with Timer() as t_inf:
    y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f'Inference time: {t_inf}')

# --- METRICS ---
best_f1, best_threshold = custom_f1_score_ml(
    y_true=test_masks,
    y_pred_proba=y_pred_proba,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    img_size=IMG_SIZE,
    n_thresholds=cfg['n_thresholds']
)

y_pred = (y_pred_proba >= best_threshold).astype(int)
y_pred_recon = reconstruct_from_patches(y_pred, BATCH_SIZE, IMG_SIZE, PATCH_SIZE)

metrics = compute_metrics(test_masks, y_pred_recon.astype(bool))
metrics['f1'] = best_f1
metrics['threshold'] = best_threshold
metrics['train_time_seconds'] = t_train.elapsed
metrics['inference_time_seconds'] = t_inf.elapsed
metrics['dataset'] = cfg['dataset']
metrics['method'] = METHOD

print(f'\nResults:')
print(f'  Precision : {metrics["precision"]:.4f}')
print(f'  Recall    : {metrics["recall"]:.4f}')
print(f'  F1        : {metrics["f1"]:.4f}')
print(f'  Threshold : {metrics["threshold"]:.4f}')
print(f'  Train time: {metrics["train_time_seconds"]:.2f}s')

# --- SAVE ---
os.makedirs(os.path.dirname(cfg['results_path']), exist_ok=True)
pd.DataFrame([metrics]).to_csv(cfg['results_path'], index=False)
print(f'\nResults saved to {cfg["results_path"]}')

# --- PLOT ---
rfi_patches = np.where(np.any(test_masks > 0, axis=(1, 2, 3)))[0]
sample_idx = rfi_patches[0] if len(rfi_patches) > 0 else 0

plot_detection_result(
    spectrogram=test_data[sample_idx],
    ground_truth=test_masks[sample_idx],
    predicted_mask=y_pred_recon[sample_idx],
    title=f'{METHOD.upper()} {cfg["dataset"].upper()} — threshold={best_threshold:.2f}',
    save_path=cfg['plot_path']
)