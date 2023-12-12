from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from itertools import product

from .ModelBase import *
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from utils import relevant_cols_from_cv_results


class LassoRegression(ModelBase):
    def _get_model(self) -> Lasso:
        return Lasso()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Lasso:
        def train_rfe(_n_features: int, _alpha: float):
            print(f"Training alpha={_alpha} with n={_n_features}")
            self._model = Lasso(alpha=_alpha)
            rfe = RFE(self._model, n_features_to_select=_n_features, verbose=1)

            rfe.fit(x_standardized, y_train)
            print(f'Finished RFE for alpha={_alpha}, n={_n_features}')
            selected_features = np.where(rfe.support_)[0]
            x_selected = x_standardized[:, selected_features]

            cv_score = cross_val_score(self._model, x_selected, y_train, n_jobs=10)
            return _n_features, _alpha, selected_features, cv_score.mean(), cv_score.std()

        x_standardized = StandardScaler().fit_transform(x_train)

        parallel = Parallel(n_jobs=10)
        all_n_features = [30, 32, 35, 37, 40]
        all_alpha = [0.03, 0.05, 0.1]
        results = list(
            parallel(delayed(train_rfe)(n_features, alpha) for n_features, alpha in product(all_n_features, all_alpha)))
        self._cv_results = pd.DataFrame(results,
                                        columns=['param_n_features', 'param_alpha', 'TO_DROP', 'avg_test_score',
                                                 'std_test_score']).drop('TO_DROP', axis=1)

        best_n_features, best_alpha, best_features, best_cv_score, best_cv_std = max(results, key=(lambda x: x[3]))
        print(f'Best: alpha={best_alpha}, n_features={best_n_features}')
        self._selected_features = x_train.columns[best_features]
        self._model.fit(x_train.loc[:, self._selected_features], y_train)
        return self._model

    def predict(self, data: pd.DataFrame) -> float:
        data_standardized = StandardScaler().fit_transform(data)
        if self._selected_features:
            return self._model.predict(data_standardized.loc[:, self._selected_features])[0]
        else:
            return self._model.predict(data_standardized)[0]
