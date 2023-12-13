from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import mean_squared_error

from DataSetType import DataSetType

_df = pd.DataFrame


class ModelBase(ABC):
    def __init__(self, dataset: (_df, _df, _df, _df), dataset_type: DataSetType):
        self._cv_results = None
        self._model = self._get_model()
        self.dataset_type = dataset_type

        x_train, x_test, y_train, y_test = dataset
        self._selected_features = x_train.columns
        self._model = self._train(x_train, y_train)

        self.test_mse = self._get_mse_test_err(x_test, y_test)

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    def _get_mse_test_err(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        y_pred = self._model.predict(x_test.loc[:, self._selected_features])
        return mean_squared_error(y_pred, y_test)

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> float:

        if self._selected_features:
            return self._model.predict(data.loc[:, self._selected_features])[0]
        else:
            return self._model.predict(data)[0]

    @property
    def cv_results(self) -> pd.DataFrame:
        return self._cv_results
