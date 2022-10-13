import json
import os
import shutil

import fire
import joblib
import mlflow
import torch

from gcnns import dataset, deep_models_utils, gcnn, unet, utils

FTP_PATH = '<TOKEN_HERE>'


def get_mlflow_artifacts_path(
        run_name: str,
        experiment_id: str,
        artifacts_storage_path: str) -> str:
    client = mlflow.tracking.MlflowClient(
        '<CLIENT_HERE>')
    result = client.search_runs([experiment_id])
    selected = []
    for r in result:
        if r.data.tags['mlflow.runName'] == run_name:
            selected.append(r)
    assert len(selected) == 1
    path = selected[0].info.artifact_uri
    path = path.replace(FTP_PATH, '/media')
    path = os.path.join(path + '/', artifacts_storage_path)
    return path


def run(dest_path: str,
        run_name: str,
        artifacts_storage_path: str,
        experiment_id: str,
        n_features: int,
        model_name: str,
        patch_batch: int,
        patch_size: int,
        train_fraction: float,
        neighborhood_radius: int,
        bilinear: bool = True,  # used only when model_name == unet
        seed: int = 0):
    args = locals()
    mlflow.set_tracking_uri("<TRACKING_URI_HERE>")
    mlflow.set_experiment('GCNNs')
    mlflow.start_run(run_name=f'inference-{run_name}')
    utils.log_params_to_mlflow(args)
    model_path = get_mlflow_artifacts_path(
        run_name,
        experiment_id, artifacts_storage_path)
    _, _, test_generator = \
        dataset.create_datasets(
            'datasets/train-eopatches/eopatches/train',
            n_features,
            patch_size, model_name, seed,
            train_fraction, patch_batch,
            neighborhood_radius, 1, True)

    os.makedirs(dest_path, exist_ok=True)

    device = torch.device('cuda')
    metrics = utils.get_metrics_dict()

    if model_name == 'gcnn':
        model = gcnn.GCN(n_features, 1)
    elif model_name == 'unet':
        model = unet.UNet(n_features, bilinear)
    elif model_name == 'rf':
        model = joblib.load(os.path.join(model_path, 'model.joblib.pkl'))
    else:
        raise ValueError('Invalid value for the model_name argument.')

    if model_name != 'rf':
        model.load_state_dict(torch.load(os.path.join(model_path, 'model')))
        model = model.to(device)
        deep_models_utils.deep_model_loop(
            model, model_name, test_generator,
            None, 'test', metrics, device)
    else:
        utils.rf_loop(model, test_generator, model_path, 'test', metrics)

    with open(os.path.join(dest_path, 'output_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile,  indent=4, sort_keys=True)

    mlflow.log_artifacts(dest_path, artifact_path=dest_path)
    shutil.rmtree(dest_path)


if __name__ == '__main__':
    fire.Fire(run)
