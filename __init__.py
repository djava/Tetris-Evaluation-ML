from constants import *
import os
import pickle
from models.ModelBase import ModelBase
from server import run_server

models: dict[ModelType, ModelBase] = {}


def init_models():
    global models
    for model_type, path in MODEL_PATHS.items():
        model_creation_fn = MODEL_TYPE_ENUM_TO_CREATION_FN[model_type]
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model = model_creation_fn(DATASET_PATH)
            with open(path, 'wb+') as f:
                pickle.dump(model, f)
            print(f'Created and saved {model_type} model at {os.path.basename(path)}',
                  f'Test MSE: {round(model.test_mse, 2)}',
                  sep=' | ')
            if model.cv_results is not None:
                csv_path = f'{os.path.dirname(path)}/cv_results_{model_type}.csv'
                model.cv_results.to_csv(csv_path, index=False)
                print(f'Saved CV results from {model_type} to {csv_path}')
            models[model_type] = model
        else:
            with open(path, 'rb') as f:
                models[model_type] = pickle.load(f)
            print(f'Loaded model: {os.path.basename(path)}')


if __name__ == '__main__':
    init_models()
    run_server(models)
