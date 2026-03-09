import numpy as np


def compute_metrics(mask_true: np.ndarray, mask_pred: np.ndarray) -> dict:
    """
    Computes binary classification metrics for RFI detection.

    Args:
        mask_true: Boolean ground truth array of any shape.
        mask_pred: Boolean predicted mask array, same shape as mask_true.

    Returns:
        dict with keys: precision, recall, f1, TP, FP, FN, TN.
    """
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    TP = np.sum(mask_pred & mask_true)
    FP = np.sum(mask_pred & ~mask_true)
    FN = np.sum(~mask_pred & mask_true)
    TN = np.sum(~mask_pred & ~mask_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN)
    }


def custom_f1_score_ml(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    batch_size: int,
    patch_size: int = 8,
    img_size: int = 512,
    n_thresholds: int = 18
) -> tuple:
    """
    Computes the best F1 score over a range of thresholds for patch-based ML methods.
    Reconstructs full images from patch predictions before computing metrics.

    Args:
        y_true: Ground truth mask of shape (batch_size, img_size, img_size, 1).
        y_pred_proba: Predicted probabilities of shape (n_patches,).
        batch_size: Number of original images.
        patch_size: Size of each patch. Default is 8.
        img_size: Size of the original image. Default is 512.
        n_thresholds: Number of thresholds to evaluate. Default is 18.

    Returns:
        Tuple of (best_f1, best_threshold).
    """
    from methods.ml.features import reconstruct_from_patches

    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_pred_recon = reconstruct_from_patches(y_pred, batch_size, img_size, patch_size)

        TP = np.sum((y_true == 1) & (y_pred_recon == 1))
        FP = np.sum((y_true == 0) & (y_pred_recon == 1))
        FN = np.sum((y_true == 1) & (y_pred_recon == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_f1, best_threshold