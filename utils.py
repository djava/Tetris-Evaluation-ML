import math
from sys import float_info

import pandas as pd
import sklearn.utils
from sklearn.model_selection import train_test_split
import numpy as np
from DataSetTypes import DataSetNorm


def normalize_sr_eval(y_i: float) -> float:
    j = 0.01
    numerator = 1.4
    denominator = 1 + math.exp(-j * (y_i + 25))
    return numerator / denominator


def inverse_normalized_eval(y_i: float | np.ndarray) -> float | np.ndarray:
    y_i = np.minimum(np.maximum(y_i, float_info.epsilon), 1)
    j = 0.01
    return ((1 / j) * -np.log((1.4 / y_i) - 1)) - 25


def get_data_split(dataset_path: str, dataset_norm: DataSetNorm) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param dataset_path: Path to get the dataset from
    :param normalized: Whether to run the SR values through the normalization function
    :return: Dataset split into x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(dataset_path)

    x = df.drop(['eval', 'Unnamed: 0'], axis=1)
    y = df['eval']
    if dataset_norm is DataSetNorm.NORMALIZED:
        y = y.apply(normalize_sr_eval)

    # Split the data into training and testing sets
    # Format: x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=0.2)


def unimplemented():
    raise Exception("Unimplemented!")


def relevant_cols_from_cv_results(cv_results: sklearn.utils.Bunch) -> pd.DataFrame:
    ret = pd.DataFrame()
    col_names = [i for i in cv_results.keys() if i.startswith('param_')] + ['mean_test_score', 'std_test_score']
    for i in col_names:
        ret[i] = cv_results[i]
    return ret


def generate_ptp_terms(heights: list[int]) -> pd.DataFrame:
    num_columns = 10
    df = pd.DataFrame({f'col{i}': heights[i] for i in range(num_columns)}, index=[0])
    for width in range(2, num_columns + 1):
        for i in range(num_columns - width + 1):
            key_name = f'ptp({",".join(f"col{j}" for j in range(i, i + width))})'
            col_slice = heights[i:i + width]
            df[key_name] = max(col_slice) - min(col_slice)

    return df
