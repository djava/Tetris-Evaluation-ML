import enum, os
from models import LinearRegression, DecisionTree, LassoRegression, RidgeRegression


class ModelType(enum.StrEnum):
    LINEAR_REGRESSION = enum.auto()
    DECISION_TREE = enum.auto()
    LASSO_REGRESSION = enum.auto()
    RIDGE_REGRESSION = enum.auto()


MODEL_PATHS = {
    ModelType.LINEAR_REGRESSION: './model_pkls/lin_reg_model.pkl',
    ModelType.DECISION_TREE: './model_pkls/dec_trees_model.pkl',
    ModelType.LASSO_REGRESSION: './model_pkls/lasso_model.pkl',
    ModelType.RIDGE_REGRESSION: './model_pkls/ridge_model.pkl'
}

MODEL_TYPE_ENUM_TO_CREATION_FN = {
    ModelType.LINEAR_REGRESSION: LinearRegression.LinearRegression,
    ModelType.DECISION_TREE: DecisionTree.DecisionTree,
    ModelType.LASSO_REGRESSION: LassoRegression.LassoRegression,
    ModelType.RIDGE_REGRESSION: RidgeRegression.RidgeRegression
}

DATASET_PATH = './labelled_placements_local_ptp.csv'

SERVER_PORT = int(os.getenv('PORT', 8000))