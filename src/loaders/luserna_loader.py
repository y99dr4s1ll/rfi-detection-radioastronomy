import pandas as pd
import numpy as np
import warnings
import os

warnings.simplefilter('ignore', np.exceptions.RankWarning)


def load_luserna(name: str, path: str, powers: bool = False) -> pd.DataFrame:
    """
    Loads a Luserna spectrogram from an HDF5 file.

    Args:
        name: Name of the file without extension (e.g. 'PG01').
        path: Directory containing the .h5 file.
        powers: If True, converts from dB to linear power. Default is False.

    Returns:
        pd.DataFrame: Spectrogram with time axis as index (in seconds).
    """
    filepath = os.path.join(path, f'{name}.h5')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Spectrogram file not found: {filepath}')

    data = pd.read_hdf(filepath)
    data.index = data.index.astype(int) / 1_000_000

    if powers:
        data = (10 ** (data / 10)) * 32768

    return data


def load_luserna_truth(path: str, file: str = 'truth_flag.h5') -> pd.DataFrame:
    """
    Loads the manually annotated RFI ground truth for the Luserna dataset.

    Args:
        path: Directory containing the ground truth file.
        file: Filename of the ground truth HDF5 file. Default is 'truth_flag.h5'.

    Returns:
        pd.DataFrame: Boolean mask with same shape as the spectrogram.
    """
    filepath = os.path.join(path, file)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Ground truth file not found: {filepath}')

    return pd.read_hdf(filepath, key='bool_data')
