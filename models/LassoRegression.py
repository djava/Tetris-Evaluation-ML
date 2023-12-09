from sklearn.linear_model import Lasso
from .ModelBase import *
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils import relevant_cols_from_cv_results


class LassoRegression(ModelBase):
    def _get_model(self) -> Lasso:
        return Lasso()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Lasso:
        param_grid = {'alpha': np.logspace(-4, 2, 20)}
        grid_search = GridSearchCV(self._model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)

        self._cv_results = relevant_cols_from_cv_results(grid_search.cv_results_)
        self._model = grid_search.best_estimator_
        return self._model
