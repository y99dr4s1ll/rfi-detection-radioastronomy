import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_rf(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = None,
    bootstrap: bool = True
) -> RandomForestClassifier:
    """
    Builds a RandomForestClassifier with the given parameters.

    Args:
        n_estimators: Number of trees. Default is 100.
        max_depth: Maximum depth of the tree. Default is None.
        min_samples_split: Minimum samples to split a node. Default is 2.
        min_samples_leaf: Minimum samples at a leaf node. Default is 1.
        max_features: Number of features for best split. Default is None.
        bootstrap: Whether to use bootstrap samples. Default is True.

    Returns:
        RandomForestClassifier: Untrained model.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight='balanced',
        random_state=42
    )


def train_rf(model: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier.

    Args:
        model: RandomForestClassifier instance.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    print('Training Random Forest...')
    model.fit(X_train, y_train)
    return model