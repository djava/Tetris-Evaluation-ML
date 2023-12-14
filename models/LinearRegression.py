import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as skLinearRegression
from .ModelBase import *
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


class LinearRegression(ModelBase):
    def _get_model(self) -> skLinearRegression:
        return skLinearRegression()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> skLinearRegression:

        # Set up cross-validation
        num_features = x_train.shape[1]
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_results = []
        best_mse = float('inf')
        best_num_features = num_features

        # Backward stepwise model selection with cross-validation
        for i in range(num_features, 30, -1):
            print(f'Fitting {i} features')

            args_sorted = np.argsort(np.abs(self._model.fit(x_train, y_train).coef_))
            selected_features = args_sorted[-i:]
            if i < num_features:
                removed_feature = x_train.columns[args_sorted[-i-1]]
            else:
                removed_feature = None
            x_selected = x_train.loc[:, x_train.columns[selected_features]]

            # Compute cross-validated mean squared error
            mse_scores = -cross_val_score(self._model, x_selected, y_train, cv=cv, scoring='neg_mean_squared_error')
            avg_mse = np.mean(mse_scores)

            # Update the best model if the current one is better
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_num_features = i
            cv_results.append((i, np.mean(mse_scores), np.std(mse_scores), removed_feature))

        self._cv_results = pd.DataFrame(cv_results, columns=['param_num_features', 'mean_mse_score', 'std_mse_score',
                                                             'removed_feature'])

        self._model.fit(x_train, y_train)
        selected_feature_idxs = np.argsort(np.abs(self._model.coef_))[-best_num_features:]
        self._selected_features = x_train.columns[selected_feature_idxs]
        print(f'Best feature count: {len(selected_feature_idxs)}')
        self._model.fit(x_train.loc[:, self._selected_features], y_train)
        return self._model

    def predict(self, data: pd.DataFrame) -> float:
        return self._model.predict(data.loc[:, self._selected_features])[0]
