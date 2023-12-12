import numpy as np
from sklearn.linear_model import LinearRegression as skLinearRegression
from .ModelBase import *
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


class LinearRegression(ModelBase):
    def _get_model(self) -> skLinearRegression:
        return skLinearRegression()

    def _train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> skLinearRegression:
        # rfe = RFE(self._model)
        #
        # param_grid = {
        #     'n_features_to_select': np.linspace(10, 55, 10, dtype='int'),
        # }
        #
        # grid_search = GridSearchCV(estimator=rfe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5,
        #                            verbose=2)
        #
        # grid_search.fit(x_train, y_train)
        #
        # self._model = grid_search.best_estimator_

        # Set up cross-validation
        num_features = x_train.shape[1]
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Backward stepwise model selection with cross-validation
        best_mse = float('inf')
        best_num_features = num_features

        for i in range(num_features, 30, -1):
            print(f'Fitting {i} features')

            selected_features = np.argsort(np.abs(self._model.fit(x_train, y_train).coef_))[-i:]
            x_selected = x_train.loc[:, x_train.columns[selected_features]]

            # Compute cross-validated mean squared error
            mse_scores = -cross_val_score(self._model, x_selected, y_train, cv=cv, scoring='neg_mean_squared_error')
            avg_mse = np.mean(mse_scores)

            # Update the best model if the current one is better
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_num_features = i

        self._model.fit(x_train, y_train)
        selected_feature_idxs = np.argsort(np.abs(self._model.coef_))[-best_num_features:]
        self._selected_features = x_train.columns[selected_feature_idxs]
        print(f'Best feature count: {len(selected_feature_idxs)}')
        self._model.fit(x_train.loc[:, self._selected_features], y_train)
        return self._model

    def predict(self, data: pd.DataFrame) -> float:
        return self._model.predict(data.loc[:, self._selected_features])[0]