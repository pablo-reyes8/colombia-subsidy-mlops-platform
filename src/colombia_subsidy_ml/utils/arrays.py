import numpy as np


def to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)
