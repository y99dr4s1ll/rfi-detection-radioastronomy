import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def build_knn(
    n_neighbors: int = 100,
    weights: str = 'distance',
    p: int = 2,
    leaf_size: int = 100,
    algorithm: str = 'auto'
) -> KNeighborsClassifier:
    """
    Builds a KNeighborsClassifier with the given parameters.

    Args:
        n_neighbors: Number of neighbors. Default is 100.
        weights: Weight function. Default is 'distance'.
        p: Power parameter for Minkowski metric. Default is 2.
        leaf_size: Leaf size for BallTree or KDTree. Default is 100.
        algorithm: Algorithm for nearest neighbor search. Default is 'auto'.

    Returns:
        KNeighborsClassifier: Untrained model.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        leaf_size=leaf_size,
        algorithm=algorithm
    )


def train_knn(model: KNeighborsClassifier, X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """
    Trains a KNeighborsClassifier.

    Args:
        model: KNeighborsClassifier instance.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        KNeighborsClassifier: Trained model.
    """
    print('Training KNN...')
    model.fit(X_train, y_train)
    return model