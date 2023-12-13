from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from .ModelBase import *
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from DataSetType import DataSetType

class LassoRegression(ModelBase):
    def _get_model(self) -> Lasso:
        return Lasso()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Lasso:
        x_standardized = StandardScaler().fit_transform(x_train)

        def train_model(_alpha: float):
            print(f"Training alpha={_alpha}")
            self._model = Lasso(alpha=_alpha)

            cv_score = cross_val_score(self._model, x_standardized, y_train, n_jobs=10)
            return _alpha, cv_score.mean(), cv_score.std()

        parallel = Parallel(n_jobs=10)
        if self.dataset_type is DataSetType.NORMALIZED:
            all_alpha = [1e-4, 5e-4, 1e-3]
        else:
            all_alpha = [0.025, 0.05, 0.075, 0.1, 0.5, 1]
        results = list(parallel(delayed(train_model)(alpha) for alpha in all_alpha))
        self._cv_results = pd.DataFrame(results, columns=['param_alpha', 'avg_test_score', 'std_test_score'])

        best_alpha, best_cv_score, best_cv_std = max(results, key=(lambda x: x[1]))
        self._model = Lasso(alpha=best_alpha)
        self._model.fit(x_train, y_train)

        print(f'Best: alpha={best_alpha}, non-zeroes={np.count_nonzero(self._model.coef_)}')
        return self._model

    def predict(self, data: pd.DataFrame) -> float:
        data_standardized = pd.DataFrame(StandardScaler().fit_transform(data.loc[:, self._selected_features]),
                                         columns=self._selected_features)
        return self._model.predict(data_standardized)[0]
