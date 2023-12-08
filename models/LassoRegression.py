from sklearn.linear_model import LassoCV
from .ModelBase import *


class LassoRegression(ModelBase):
    def _get_model(self) -> LassoCV:
        return LassoCV()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> LassoCV:
        self._model.fit(x_train, y_train)
        return self._model
