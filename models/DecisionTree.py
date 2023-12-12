from sklearn.tree import DecisionTreeRegressor

from utils import relevant_cols_from_cv_results
from .ModelBase import *
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE


class DecisionTree(ModelBase):
    def _get_model(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> DecisionTreeRegressor:
        # Cross-validate over different hyperparameters
        # rfe = RFE(self._model)
        #
        # param_grid = {
        #     'n_features_to_select': np.linspace(10, 55, 10, dtype='int'),
        # }

        param_grid = {
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=self._model, param_grid=param_grid, cv=5,
                                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(x_train, y_train)

        self._cv_results = relevant_cols_from_cv_results(grid_search.cv_results_)

        self.model = grid_search.best_estimator_
        return self.model

    def predict(self, data: pd.DataFrame) -> float:
        return self._model.predict(data)[0]