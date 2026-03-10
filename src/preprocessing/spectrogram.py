import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def polynomial_detrend(data: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Removes a polynomial trend from the spectrogram along the time axis.

    Args:
        data: 2D array of shape (time, frequency).
        degree: Degree of the polynomial. Default is 2.

    Returns:
        np.ndarray: Detrended data with the same shape as input.
    """
    x = np.arange(data.shape[1]).reshape(-1, 1)
    y = np.mean(data, axis=0)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(poly_features, y)
    trend = model.predict(poly_features)

    return data - trend



def extract_and_split_patches(
    data: np.ndarray,
    masks: np.ndarray,
    patch_size: tuple = (512, 512),
    train_size: int = 492,
    max_patches: int = 612,
    random_seed: int = 42
) -> tuple:
    patches = extract_patches_2d(data, patch_size, max_patches=max_patches, random_state=random_seed)
    patches = patches[..., np.newaxis]
    patch_masks = extract_patches_2d(masks, patch_size, max_patches=max_patches, random_state=random_seed)
    patch_masks = patch_masks[..., np.newaxis]

    indices = np.arange(patches.shape[0])
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    return (
        patches[train_idx],
        patch_masks[train_idx],
        patches[test_idx],
        patch_masks[test_idx]
    )

def balance_dataset(
    data: np.ndarray,
    masks: np.ndarray,
    random_seed: int = 42
) -> tuple:
    """
    Removes patches with no RFI to reduce class imbalance.
    Keeps all positive patches (containing at least one flagged pixel)
    and discards all negative patches.

    Args:
        data: Array of patches of shape (n, h, w, 1).
        masks: Boolean array of shape (n, h, w, 1).
        random_seed: Random seed for reproducibility. Default is 42.

    Returns:
        Tuple of (balanced_data, balanced_masks).
    """
    has_rfi = np.any(masks > 0, axis=(1, 2, 3))

    positive_idx = np.where(has_rfi)[0]
    selected_idx = positive_idx

    rng = np.random.default_rng(random_seed)
    shuffle_idx = rng.permutation(len(selected_idx))

    return data[selected_idx[shuffle_idx]], masks[selected_idx[shuffle_idx]]
    
def get_patches_batched(data: np.ndarray, p_size: tuple, s_size: tuple, rate: tuple, padding: str = 'VALID', batch_size: int = 50) -> np.ndarray:
    """
    Applies get_patches in batches to avoid GPU memory overflow.

    Args:
        data: Array of shape (n, h, w, 1).
        p_size: Patch size tuple.
        s_size: Stride size tuple.
        rate: Dilation rate tuple.
        padding: Padding mode. Default is 'VALID'.
        batch_size: Number of images per batch. Default is 50.

    Returns:
        np.ndarray: Concatenated patches.
    """
    from utils.data import get_patches
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        patches = get_patches(batch, None, p_size, s_size, rate, padding)
        results.append(patches)
    return np.concatenate(results, axis=0)