from sklearn.linear_model import Ridge
from .ModelBase import *
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils import relevant_cols_from_cv_results


class RidgeRegression(ModelBase):
    def _get_model(self) -> Ridge:
        return Ridge()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Ridge:
        param_grid = {'alpha': np.logspace(-4, 2, 20)}
        grid_search = GridSearchCV(self._model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)

        self._cv_results = relevant_cols_from_cv_results(grid_search.cv_results_)
        self._model = grid_search.best_estimator_
        return self._model
