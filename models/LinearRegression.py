from sklearn.linear_model import LinearRegression as skLinearRegression
from .ModelBase import *


class LinearRegression(ModelBase):
    def _get_model(self) -> skLinearRegression:
        return skLinearRegression()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> skLinearRegression:
        self._model.fit(x_train, y_train)
        return self._model
