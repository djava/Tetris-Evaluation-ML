import pandas as pd
import sklearn.utils
from sklearn.model_selection import train_test_split
from functools import reduce
import numpy as np


def add_local_interaction_columns(data: pd.DataFrame) -> pd.DataFrame:
    num_columns = 10
    for width in range(2, num_columns + 1):
        for i in range(num_columns - width + 1):
            key_name = f'ptp({",".join(f"col{j}" for j in range(i, i + width))})'
            print(f'Adding column: {key_name}')
            col = data.loc[:, (f'col{j}' for j in range(i, i + width))].apply(np.ptp, axis=1)

            data[key_name] = col
    data.to_csv('labelled_placements_local_ptp.csv')
    return data


def get_data_split(dataset_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param dataset_path: Path to get the dataset from
    :return: Dataset split into x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(dataset_path)
    # df = add_local_interaction_columns(df)

    x = df.drop('eval', axis=1)
    y = df['eval']

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
    df = pd.Dataframe({f'col{i}': heights[i] for i in range(num_columns)})
    for width in range(2, num_columns + 1):
        for i in range(num_columns - width + 1):
            key_name = f'ptp({",".join(f"col{j}" for j in range(i, i + width))})'
            col_slice = heights[i:i + width]
            df[key_name] = max(col_slice) - min(col_slice)

    return df
