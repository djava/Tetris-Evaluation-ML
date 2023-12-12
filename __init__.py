import utils
from constants import *
import os
import pickle
from models.ModelBase import ModelBase
from server import run_server

models: dict[ModelType, ModelBase] = {}
FORCE_RETRAIN: list[ModelType] = []

dataset = utils.get_data_split(DATASET_PATH)


def init_models():
    global models
    for model_type, model_path in MODEL_PATHS.items():
        model_creation_fn = MODEL_TYPE_ENUM_TO_CREATION_FN[model_type]
        if not os.path.isfile(model_path) or model_type in FORCE_RETRAIN:
            print(f'Training new {model_type} model...')

            models_dir = os.path.dirname(model_path)
            archive_dir = f'{models_dir}/archive'
            os.makedirs(archive_dir, exist_ok=True)  # Creates models_dir on the way
            csv_path = f'{models_dir}/cv_results_{model_type}.csv'

            model = model_creation_fn(dataset)

            # Archive old model if we're forcing a retrain
            if model_type in FORCE_RETRAIN and os.path.isfile(model_path):
                with open(model_path, 'rb') as f:
                    old_model = pickle.load(f)
                timestamp = os.path.getmtime(model_path)
                new_pkl_suffix = f'-mse{int(old_model.test_mse)}-ts{int(timestamp)}.pkl'
                new_pkl_name = os.path.basename(model_path).replace('.pkl', new_pkl_suffix)
                os.rename(model_path, f'{archive_dir}/{new_pkl_name}')
                print(f'Archived old model as {new_pkl_name}')

                if os.path.isfile(csv_path):
                    new_csv_suffix = new_pkl_suffix.replace('.pkl', '.csv')
                    new_csv_name = os.path.basename(csv_path).replace('.csv', new_csv_suffix)
                    os.rename(csv_path, f'{archive_dir}/{new_csv_name}')
                    print(f'Archived old CV results as {new_csv_name}')

            with open(model_path, 'wb+') as f:
                pickle.dump(model, f)

            print(f'Created and saved {model_type} model at {os.path.basename(model_path)}',
                  f'Test MSE: {round(model.test_mse, 2)}',
                  sep=' | ')

            if model.cv_results is not None:
                model.cv_results.to_csv(csv_path, index=False)
                print(f'Saved CV results from {model_type} to {csv_path}')
            models[model_type] = model
        else:
            with open(model_path, 'rb') as f:
                models[model_type] = pickle.load(f)
            print(f'Loaded model: {os.path.basename(model_path)}')


if __name__ == '__main__':
    init_models()
    run_server(models)
