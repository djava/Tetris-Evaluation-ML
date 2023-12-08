from sklearn.linear_model import RidgeCV
from .ModelBase import *


class RidgeRegression(ModelBase):
    def _get_model(self) -> RidgeCV:
        return RidgeCV()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> RidgeCV:
        self._model.fit(x_train, y_train)
        return self._model
