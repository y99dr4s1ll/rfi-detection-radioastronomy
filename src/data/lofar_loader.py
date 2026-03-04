from utils.data import get_lofar_data


class Args:
    def __init__(self):
        self.model = 'UNET'
        self.anomaly_class = 1
        self.latent_dim = 32
        self.data = 'LOFAR'
        self.data_path = None
        self.seed = 42
        self.patches = True
        self.patch_x = 32
        self.patch_y = 32
        self.patch_stride_x = 32
        self.patch_stride_y = 32
        self.input_shape = (512, 512, 1)

    def update_input_shape(self):
        if self.patches:
            self.input_shape = (self.patch_x, self.patch_y, self.input_shape[-1])


def load_lofar(data_path: str, **kwargs) -> tuple:
    """
    Loads the LOFAR dataset using RFI-NLN's get_lofar_data.

    Args:
        data_path: Path to the LOFAR pickle file directory.
        **kwargs: Optional overrides for Args fields (e.g. patch_x=64).

    Returns:
        tuple: (train_data, train_masks, test_data, test_masks)
    """
    args = Args()
    args.data_path = data_path
    for key, value in kwargs.items():
        setattr(args, key, value)
    args.update_input_shape()

    return get_lofar_data(args)