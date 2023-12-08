from utils import get_data_split
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import mean_squared_error


class ModelBase(ABC):
    def __init__(self, dataset_path: str):
        self._cv_results = None
        self._model = self._get_model()
        x_train, x_test, y_train, y_test = get_data_split(dataset_path)
        self._model = self._train(x_train.values, y_train.values)

        self.test_mse = self._get_mse_test_err(x_test, y_test)

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    def _get_mse_test_err(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        y_pred = self._model.predict(x_test.values)
        return mean_squared_error(y_pred, y_test)

    def predict(self, heights: list[int]) -> float:
        return self._model.predict([heights])[0]

    @property
    def cv_results(self) -> pd.DataFrame:
        return self._cv_results
