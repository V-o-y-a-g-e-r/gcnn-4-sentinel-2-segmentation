import json
import os
import shutil
import warnings
from ctypes import ArgumentError

import fire
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning

from gcnns import dataset, deep_models_utils, gcnn, unet, utils

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore',
                        category=UndefinedMetricWarning)


def run(dest_path: str, train_fraction: float, model_name: str,
        n_features: int = 10, epochs: int = 1, img_batch: int = 1,
        patch_batch: int = 200, patch_size: int = 10,
        neighborhood_radius: int = 1, patience: int = 3,
        seed: int = 0, learning_rate: float = 0.001, shuffle: bool = True,
        verbose: int = 0, load_checkpoint: bool = False,
        use_mlflow: bool = False, run_name: str = 'default_run',
        data_path: str = 'datasets/train-eopatches/eopatches/train') -> None:
    if use_mlflow:
        args = locals()
        mlflow.set_tracking_uri("<URI_HERE>")
        mlflow.set_experiment('GCNNs')
        mlflow.start_run(run_name=run_name)
        utils.log_params_to_mlflow(args)

    train_generator, val_generator, test_generator = \
        dataset.create_datasets(
            data_path, n_features,
            patch_size, model_name, seed,
            train_fraction, patch_batch,
            neighborhood_radius, img_batch,
            shuffle)

    os.makedirs(dest_path, exist_ok=True)
    metrics = utils.get_metrics_dict()

    if model_name == 'gcnn':
        deep_models_utils.train_deep_models(
            gcnn.GCN(n_features, 1), model_name,
            train_generator, val_generator,
            test_generator, epochs, learning_rate,
            verbose, dest_path, metrics,
            patience, load_checkpoint)
    elif model_name == 'unet':
        deep_models_utils.train_deep_models(
            unet.UNet(n_features, bilinear=True),
            model_name, train_generator, val_generator,
            test_generator, epochs, learning_rate,
            verbose, dest_path, metrics,
            patience, load_checkpoint)
    elif model_name == 'rf':
        model = RandomForestClassifier(
            random_state=seed, warm_start=True, verbose=verbose)
        utils.rf_loop(model, train_generator, dest_path, 'train', metrics)
        utils.rf_loop(model, val_generator, dest_path, 'val', metrics)
        utils.rf_loop(model, test_generator, dest_path, 'test', metrics)
        for key in metrics.keys():
            val = np.mean(metrics[key])
            metrics[key] = val
            print(f'Mean of\t{key.upper()}:\t{val:15.3f}')
    else:
        raise ArgumentError('The value of "model_name" argument is incorrect.')

    with open(os.path.join(dest_path, 'output_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile,  indent=4, sort_keys=True)

    if use_mlflow:
        mlflow.log_artifacts(dest_path, artifact_path=dest_path)
        shutil.rmtree(dest_path)


if __name__ == '__main__':
    fire.Fire(run)
