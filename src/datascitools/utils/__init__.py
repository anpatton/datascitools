import pandas as pd
import pickle
import pathlib
import numpy as np
from sklearn.metrics import mean_squared_error
from jenkspy import jenks_breaks
from numpy.typing import ArrayLike
from itertools import chain

def cbind(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """R clone for column binding"""
    return pd.concat([df1.reset_index(drop=True), df2], axis=1)

def read_pickle(file: str):
    """Read pickle object from file."""
    return pickle.load(open(file, "rb"))

def write_pickle(object, file: str) -> bool:
    """Write pickle object to file."""
    file_path = pathlib.Path.cwd() / f"{file}"
    with file_path.open("wb") as f:
        pickle.dump(object, f)
    return True

def rmse(pred: ArrayLike, actual: ArrayLike, digits: int = 3) -> float:
    """Rounded RMSE"""
    return np.round(mean_squared_error(pred, actual, squared=False), digits)

def jenks_discretize(data, n_classes: int) -> np.array:
    """Jenks breaks of vector"""
    breaks = jenks_breaks(data, n_classes=n_classes)
    breaks[-1] = np.inf
    return np.digitize(data, breaks)

def expand_lists(item_repeats, *lists_of_items):
    """Provide # of repetitions for each item in list(s)"""
    return [list(chain([[lists_of_items]*item_repeats for lists_of_items, item_repeats in zip(lists_of_items, item_repeats)])) for item_repeats in lists_of_items]
