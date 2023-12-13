import enum, os
from models import LinearRegression, DecisionTree, LassoRegression
from DataSetTypes import *


class ModelID(enum.StrEnum):
    LINEAR_REGRESSION_18 = enum.auto()
    LINEAR_REGRESSION_19 = enum.auto()
    LINEAR_REGRESSION_29 = enum.auto()
    DECISION_TREE_18 = enum.auto()
    DECISION_TREE_19 = enum.auto()
    DECISION_TREE_29 = enum.auto()
    LASSO_REGRESSION_18 = enum.auto()
    LASSO_REGRESSION_19 = enum.auto()
    LASSO_REGRESSION_29 = enum.auto()


class ModelType(enum.Enum):
    LINEAR_REGRESSION = enum.auto()
    DECISION_TREE = enum.auto()
    LASSO_REGRESSION = enum.auto()


MODEL_PATHS: dict[ModelID, str] = {
    ModelID.LINEAR_REGRESSION_18: './model_pkls/lin_reg18_model.pkl',
    ModelID.LINEAR_REGRESSION_19: './model_pkls/lin_reg19_model.pkl',
    ModelID.LINEAR_REGRESSION_29: './model_pkls/lin_reg29_model.pkl',
    ModelID.DECISION_TREE_18: './model_pkls/dec_trees18_model.pkl',
    ModelID.DECISION_TREE_19: './model_pkls/dec_trees19_model.pkl',
    ModelID.DECISION_TREE_29: './model_pkls/dec_trees29_model.pkl',
    ModelID.LASSO_REGRESSION_18: './model_pkls/lasso18_model.pkl',
    ModelID.LASSO_REGRESSION_19: './model_pkls/lasso19_model.pkl',
    ModelID.LASSO_REGRESSION_29: './model_pkls/lasso29_model.pkl',
}

MODEL_TYPE_TO_TYPE_OBJ: dict[ModelType, type] = {
    ModelType.LINEAR_REGRESSION: LinearRegression.LinearRegression,
    ModelType.DECISION_TREE: DecisionTree.DecisionTree,
    ModelType.LASSO_REGRESSION: LassoRegression.LassoRegression
}

MODEL_INFO: dict[ModelID, (type, DataSetNorm, DataSetLevel)] = {
    ModelID.LINEAR_REGRESSION_18: (ModelType.LINEAR_REGRESSION, DataSetNorm.NORMALIZED, DataSetLevel.LEVEL_18),
    ModelID.LINEAR_REGRESSION_19: (ModelType.LINEAR_REGRESSION, DataSetNorm.NORMALIZED, DataSetLevel.LEVEL_19),
    ModelID.LINEAR_REGRESSION_29: (ModelType.LINEAR_REGRESSION, DataSetNorm.NORMALIZED, DataSetLevel.LEVEL_29),
    ModelID.DECISION_TREE_18: (ModelType.DECISION_TREE, DataSetNorm.NOT_NORMALIZED, DataSetLevel.LEVEL_18),
    ModelID.DECISION_TREE_19: (ModelType.DECISION_TREE, DataSetNorm.NOT_NORMALIZED, DataSetLevel.LEVEL_19),
    ModelID.DECISION_TREE_29: (ModelType.DECISION_TREE, DataSetNorm.NOT_NORMALIZED, DataSetLevel.LEVEL_29),
    ModelID.LASSO_REGRESSION_18: (ModelType.LASSO_REGRESSION, DataSetNorm.NORMALIZED, DataSetLevel.LEVEL_18),
    ModelID.LASSO_REGRESSION_19: (ModelType.LASSO_REGRESSION, DataSetNorm.NORMALIZED, DataSetLevel.LEVEL_19),
    ModelID.LASSO_REGRESSION_29: (ModelType.LASSO_REGRESSION, DataSetNorm.NORMALIZED, DataSetLevel.LEVEL_29),
}

DATASET_PATHS: dict[DataSetLevel, str] = {
    DataSetLevel.LEVEL_18: 'datasets/level_18_local_ptp.csv',
    DataSetLevel.LEVEL_19: 'datasets/level_19_local_ptp.csv',
    DataSetLevel.LEVEL_29: 'datasets/level_29_local_ptp.csv',
}

SERVER_PORT = int(os.getenv('PORT', 8000))
