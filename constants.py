import enum, os
from models import LinearRegression, DecisionTree, LassoRegression
from DataSetType import DataSetType


class ModelType(enum.StrEnum):
    LINEAR_REGRESSION = enum.auto()
    DECISION_TREE = enum.auto()
    LASSO_REGRESSION = enum.auto()
    LINEAR_REGRESSION_NORM = enum.auto()
    DECISION_TREE_NORM = enum.auto()
    LASSO_REGRESSION_NORM = enum.auto()


MODEL_PATHS: dict[ModelType, str] = {
    ModelType.LINEAR_REGRESSION: './model_pkls/lin_reg_model.pkl',
    ModelType.DECISION_TREE: './model_pkls/dec_trees_model.pkl',
    ModelType.LASSO_REGRESSION: './model_pkls/lasso_model.pkl',
    ModelType.LINEAR_REGRESSION_NORM: './model_pkls/lin_reg_norm_model.pkl',
    ModelType.DECISION_TREE_NORM: './model_pkls/dec_trees_norm_model.pkl',
    ModelType.LASSO_REGRESSION_NORM: './model_pkls/lasso_norm_model.pkl',
}


MODEL_TYPE_ENUM_TO_CREATION_INFO: dict[ModelType, (type, DataSetType)] = {
    ModelType.LINEAR_REGRESSION: (LinearRegression.LinearRegression, DataSetType.NOT_NORMALIZED),
    ModelType.DECISION_TREE: (DecisionTree.DecisionTree, DataSetType.NOT_NORMALIZED),
    ModelType.LASSO_REGRESSION: (LassoRegression.LassoRegression, DataSetType.NOT_NORMALIZED),
    ModelType.LINEAR_REGRESSION_NORM: (LinearRegression.LinearRegression, DataSetType.NORMALIZED),
    ModelType.DECISION_TREE_NORM: (DecisionTree.DecisionTree, DataSetType.NORMALIZED),
    ModelType.LASSO_REGRESSION_NORM: (LassoRegression.LassoRegression, DataSetType.NORMALIZED),
}

DATASET_PATH = './labelled_placements_local_ptp.csv'

SERVER_PORT = int(os.getenv('PORT', 8000))
