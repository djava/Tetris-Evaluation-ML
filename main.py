from utils import get_data_split
from constants import *
import os
import pickle
from models.ModelBase import ModelBase
from server import run_server
import numpy as np
import sys

models: dict[ModelID, ModelBase] = {}
FORCE_RETRAIN: list[ModelID | ModelType | DataSetLevel | DataSetNorm] = [ModelID.LINEAR_REGRESSION_18]


def must_retrain_model(model_id: ModelID, model_type: ModelType,
                       dataset_norm: DataSetNorm, dataset_level: DataSetLevel) -> bool:
    return 'train' in sys.argv and any(i in FORCE_RETRAIN for i in ['*', model_id, model_type,
                                                                    dataset_norm, dataset_level])


def archive_old_model(model_path: str, cv_results_csv_path: str) -> None:
    archive_dir = f'{os.path.dirname(model_path)}/archive'
    os.makedirs(archive_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        old_model = pickle.load(f)
    timestamp = os.path.getmtime(model_path)
    new_pkl_suffix = f'-mse{int(old_model.test_mse)}-ts{int(timestamp)}.pkl'
    new_pkl_name = os.path.basename(model_path).replace('.pkl', new_pkl_suffix)
    os.rename(model_path, f'{archive_dir}/{new_pkl_name}')
    print(f'Archived old model as {new_pkl_name}')

    if os.path.isfile(cv_results_csv_path):
        new_csv_suffix = new_pkl_suffix.replace('.pkl', '.csv')
        new_csv_name = os.path.basename(cv_results_csv_path).replace('.csv', new_csv_suffix)
        os.rename(cv_results_csv_path, f'{archive_dir}/{new_csv_name}')
        print(f'Archived old CV results as {new_csv_name}')


def init_models():
    global models

    for model_id in ModelID:
        model_path = MODEL_PATHS[model_id]
        model_type, dataset_norm, dataset_level = MODEL_INFO[model_id]
        if not os.path.isfile(model_path) or must_retrain_model(model_id, model_type, dataset_norm, dataset_level):
            if 'train' not in sys.argv:
                raise Exception("Models missing, but not in train mode!")

            print(f'Training new {model_id} model...')

            models_dir = os.path.dirname(model_path)
            cv_results_csv_path = f'{models_dir}/cv_results_{model_id}.csv'

            dataset = get_data_split(DATASET_PATHS[dataset_level], dataset_norm)
            model = MODEL_TYPE_TO_TYPE_OBJ[model_type](dataset, dataset_norm)

            if os.path.isfile(model_path):
                archive_old_model(model_path, cv_results_csv_path)
            else:
                os.makedirs(models_dir, exist_ok=True)

            with open(model_path, 'wb+') as f:
                pickle.dump(model, f)

            print(f'Created and saved {model_id} model at {os.path.basename(model_path)}',
                  f'Test MSE: {round(model.test_mse, 2)}',
                  sep=' | ')

            if model.cv_results is not None:
                model.cv_results.to_csv(cv_results_csv_path, index=False)
                print(f'Saved CV results from {model_id} to {cv_results_csv_path}')

            models[model_id] = model

        else:
            with open(model_path, 'rb') as f:
                models[model_id] = pickle.load(f)

            print(f'Loaded model: {os.path.basename(model_path)}')


if __name__ == '__main__':
    init_models()
    run_server(models)
