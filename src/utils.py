import pandas as pd
import pickle
import pathlib
import numpy as np
from sklearn.metrics import mean_squared_error
from jenkspy import jenks_breaks

def cbind(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """R clone for column binding"""
    return pd.concat([df1.reset_index(drop=True), df2], axis=1)

def read_pickle(file: str):
    """Read pickle object from file."""
    return pickle.load(open(file, "rb"))

def write_pickle(object, file: str) -> bool:
    """Write pickle object to file."""
    file_path = pathlib.Path.cwd()/f"{file}"
    with file_path.open("wb") as f:
        pickle.dump(object, f)
    return True

def rmse(pred, actual, digits=3) -> float:
    """Rounded RMSE"""
    return np.round(mean_squared_error(pred, actual, squared=False), digits)

def jenks_discretize(data, n_classes: int) -> np.array:
    breaks = jenks_breaks(data, n_classes=n_classes)
    breaks[-1] = np.inf
    return np.digitize(data, breaks)