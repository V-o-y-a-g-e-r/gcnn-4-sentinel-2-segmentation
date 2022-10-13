import os
import shutil
from argparse import ArgumentError
from itertools import product
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import rasterio
import torch
from rasterio.warp import calculate_default_transform

from gcnns.dataset import DataGenerator, LoadTask
from gcnns.gcnn import GCN
from gcnns.unet import UNet
from gcnns.utils import log_params_to_mlflow


def get_mlflow_artifacts_path(artifacts_storage_path: str) -> str:
    filter_string = 'parameters.artifacts_storage = \'{}\''\
        .format(artifacts_storage_path)
    result = mlflow.search_runs(filter_string=filter_string)['artifact_uri'][0]
    result = result.replace(
        '<URI_HERE>', '<PATH>')
    return os.path.join(result + '/', artifacts_storage_path)


def _export_tiff(image_array: np.ndarray, path: str, transform):
    with rasterio.open(
            path, 'w', driver='GTiff', width=2000, height=2000,
            count=1, dtype=image_array.dtype, nodata=255,
            transform=transform, crs='epsg:32633', compress=None) as dst:
        dst.write(image_array)


def run(dest_path: str, dir_path: str, patch_size: int,
        neighborhood_size: int, model_name: str, n_features: str,
        device: str = 'cuda',
        dataset_path: str = 'datasets/test-eopatches/eopatches/test'):
    args = locals()
    mlflow.set_tracking_uri("<URI_HERE>")
    mlflow.set_experiment('GCNNs')
    mlflow.start_run(
        run_name=f'inference-{model_name}-{patch_size}-{n_features}')
    log_params_to_mlflow(args)
    dir_path = get_mlflow_artifacts_path(dir_path)
    data_dir = Path(dataset_path)
    output_dir = Path(dest_path)
    os.makedirs(output_dir, exist_ok=True)
    submission_dir = output_dir / 'preds'
    vis_dir = output_dir / 'vis'
    print(submission_dir)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    examples = sorted(list(data_dir.glob('*')))
    load_task = LoadTask(path=str(data_dir))
    if model_name == 'gcnn':
        model = GCN(n_features, 1).to(device)
    elif model_name == 'unet':
        model = UNet(n_features, bilinear=False)
    else:
        raise ArgumentError("Argument 'model_name' is incorrect")
    model.load_state_dict(
        torch.load(os.path.join(dir_path, model_name),
                   map_location=device))
    dataset = DataGenerator(
        data_dir, examples, patch_batch=1,
        time_step_max=n_features, ext=neighborhood_size,
        patch_size=patch_size, model_name=model_name,
        shuffle=False)
    for img_index in range(len(dataset.samples)):
        print(img_index)
        img_patches = dataset[img_index]
        img_pred = np.zeros(shape=(2000, 2000))
        for patch_index, (i, j) in enumerate(
            product(range(0, 2000, patch_size),
                    range(0, 2000, patch_size))):
            if model_name == 'gcnn':
                y_pred = model(
                    img_patches[patch_index].x.to(device),
                    img_patches[patch_index].edge_index.to(device))
                y_pred = y_pred.cpu().detach().numpy().squeeze()\
                    .reshape(patch_size, patch_size)
            elif model_name == 'unet':
                y_pred = model(
                    img_patches[patch_index][0].unsqueeze(0)
                    .to(device)).squeeze()
                y_pred = y_pred.cpu().detach().numpy().squeeze()\
                    .reshape(patch_size, patch_size)
            else:
                raise ArgumentError(
                    "Argument 'model_name' is incorrect")
            img_pred[i:i + patch_size, j:j + patch_size] = y_pred

        example_path = dataset.samples[img_index]
        img_pred = img_pred.reshape(1, 2000, 2000)
        example = load_task.execute(eopatch_folder=example_path.name)
        y = (img_pred > 0.5).astype(np.uint8)
        dst_transform, _, _ = calculate_default_transform(
            example['bbox'].crs.ogc_string().lower(),
            'epsg:32633', 2000, 2000, *example['bbox'])
        _export_tiff(y, str(submission_dir / example_path.name) + '.tif',
                     dst_transform)
        plt.imshow(img_pred.squeeze(), cmap='gray_r')
        plt.axis('off')
        plt.savefig(os.path.join(vis_dir, f'{example_path.name}-soft.png'),
                    dpi=1000, bbox_inches='tight', pad_ionches=0)
        plt.clf()
        plt.imshow((img_pred.squeeze() > 0.5), cmap='gray_r')
        plt.axis('off')
        plt.savefig(os.path.join(vis_dir, f'{example_path.name}-hard.png'),
                    dpi=1000, bbox_inches='tight', pad_ionches=0)
        plt.clf()
    mlflow.log_artifacts(dest_path, artifact_path=dest_path)
    shutil.rmtree(dest_path)


if __name__ == '__main__':
    fire.Fire(run)
