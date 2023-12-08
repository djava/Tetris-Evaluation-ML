import pandas as pd
from sklearn.model_selection import train_test_split


def get_data_split(dataset_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param dataset_path: Path to get the dataset from
    :return: Dataset split into x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(dataset_path)

    x = df.iloc[:, :10]  # First 10 columns are columns heights
    y = df.iloc[:, -1]  # Last column is StackRabbit evaluation

    # Split the data into training and testing sets
    # Format: x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=0.2)


def unimplemented():
    raise Exception("Unimplemented!")