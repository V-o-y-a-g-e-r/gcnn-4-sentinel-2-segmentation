import random
from ctypes import ArgumentError
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from eolearn.core import LoadTask
from torch.nn import functional as F
from torch.nn.functional import adaptive_max_pool1d
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric import loader
from torch_geometric.data import Data
from tqdm import tqdm

from gcnns import dataset
from gcnns.random_forest import Features, get_features


def create_datasets(data_path: str, n_features: int, patch_size: int,
                    model_name: str, seed: int, train_fraction: float,
                    patch_batch: int, neighborhood_radius: int,
                    img_batch: int, shuffle: bool) -> Tuple[DataLoader]:
    eopatch_dir = Path(data_path)
    samples = sorted(list(eopatch_dir.glob('*')))
    print('Total number of images', len(samples))
    # Load the training data:
    ds = dataset.DataGenerator(
        model_name=model_name, shuffle=shuffle,
        data_dir=eopatch_dir, samples=samples,
        patch_batch=patch_batch, time_step_max=n_features,
        ext=neighborhood_radius, patch_size=patch_size)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    train_size = int(len(ds) * train_fraction)
    train_indices = np.random.choice(len(ds),
                                     size=train_size, replace=False)
    val_indices = np.random.choice(
        train_indices, size=int(0.05 * len(train_indices)), replace=False)
    train_indices = np.setdiff1d(train_indices, val_indices)
    # Get the test set indices:
    test_indices = np.setdiff1d(
        np.arange(len(ds)), np.concatenate((train_indices, val_indices)))
    print('Train indices: ', train_indices)
    print('Val indices: ', val_indices)
    print('Test indices: ', test_indices)
    assert np.unique(
        np.concatenate((train_indices, val_indices, test_indices))).size == \
        np.concatenate((train_indices, val_indices, test_indices)).size
    # Load the training data:
    train_generator = DataLoader(Subset(
        ds, train_indices),
        batch_size=img_batch, shuffle=shuffle,
        collate_fn=ds.custom_collate)
    # Load the validation data:
    val_generator = DataLoader(Subset(
        ds, val_indices),
        batch_size=img_batch, shuffle=shuffle,
        collate_fn=ds.custom_collate)
    # Load the test data:
    if train_fraction != 1:
        test_generator = DataLoader(Subset(
            ds, test_indices),
            batch_size=img_batch, shuffle=shuffle,
            collate_fn=ds.custom_collate)
    else:
        test_generator = []
    print(f'Training on: {len(train_generator.dataset)} images')
    print(f'Validating on: {len(val_generator.dataset)} images')
    if len(test_generator) > 0:
        print(f'Testing on: {len(test_generator.dataset)} images')

    return train_generator, val_generator, test_generator


class DataGenerator(Dataset):
    INIT_DIM = (500, 500)
    DEST_DIM = (2000, 2000)
    SCALE = 4

    def compute_stats(self) -> Dict[str, Union[int, float]]:
        time_steps = []
        for _, sample_path in enumerate(
                tqdm(self.samples, total=len(self.samples))):
            if str(sample_path).startswith('h_') or \
                    str(sample_path).startswith('v_'):
                data = self.load_task.execute(
                    eopatch_folder=Path(str(sample_path)[2:]).name)
            else:
                data = self.load_task.execute(
                    eopatch_folder=sample_path.name)
            x = data[('data', 'CLP')].astype(np.float32)
            n_time_steps = x.shape[0]
            time_steps.append(n_time_steps)
        out = {}
        out['min'] = np.min(time_steps)
        out['max'] = np.max(time_steps)
        out['mean'] = np.round(np.mean(time_steps), 3)
        out['std'] = np.round(np.std(time_steps), 3)
        out['median'] = np.median(time_steps)
        return out

    def __init__(self, data_dir: Path, samples: List, patch_batch: int,
                 time_step_max: int, ext: int, shuffle: bool,
                 patch_size: int, model_name: str):
        self.samples = samples
        self.time_step_max = time_step_max
        """
        TODO: Decide whether to use those images:
        self.samples += [Path('h_' + str(p)) for p in samples]
        self.samples += [Path('v_' + str(p)) for p in samples]
        """
        self.load_task = LoadTask(path=str(data_dir))
        self.ext = ext
        self.patch_size = patch_size
        self.adj_matrix = self.get_adj_matrix()
        self.patch_batch = patch_batch
        self.model_name = model_name
        self.features_list = list(Features)
        self.kernel_size = [1]
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.shuffle = shuffle

    def get_adj_matrix(self) -> torch.Tensor:
        m = np.arange(self.patch_size * self.patch_size) \
            .reshape(self.patch_size, self.patch_size)
        edges = []
        for i, j in product(range(m.shape[0]), range(m.shape[1])):
            left, right = max(0, j - self.ext), min(self.patch_size,
                                                    j + self.ext + 1)
            top, bottom = max(0, i - self.ext), min(self.patch_size,
                                                    i + self.ext + 1)
            neighbors = m[top:bottom, left:right].ravel().tolist()
            for n in neighbors:
                edges.append((m[i, j], n))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def __len__(self) -> int:
        return int(len(self.samples))

    def custom_collate(self, batch: List) ->\
            Union[Tuple[np.ndarray, np.ndarray], loader.DataLoader]:
        raveled_batch = []
        for sample in batch:
            raveled_batch.extend(sample)
        # The raveled_batch is now a list of torch_geometric.data.Data objects:
        if self.model_name == 'gcnn':
            raveled_batch = loader.DataLoader(
                raveled_batch, batch_size=self.patch_batch,
                shuffle=self.shuffle)
            return raveled_batch
        elif self.model_name == 'unet':
            raveled_batch = DataLoader(
                raveled_batch, batch_size=self.patch_batch,
                shuffle=self.shuffle)
            return raveled_batch
        elif self.model_name == 'rf':
            return raveled_batch
        else:
            raise ArgumentError(
                'The value of the "model_name" argument is incorrect.')

    def load_and_upscale_img(self, index: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_path = self.samples[index]
        if str(sample_path).startswith('h_') or \
                str(sample_path).startswith('v_'):
            data = self.load_task.execute(
                eopatch_folder=Path(str(sample_path)[2:]).name)
        else:
            data = self.load_task.execute(
                eopatch_folder=sample_path.name)
        # Get the data from eo-learn dict:
        try:
            y = data[('mask_timeless', 'CULTIVATED')].squeeze()
        except KeyError as error:
            print(error)
            print('No GT mask for the image')
            y = np.zeros(shape=self.DEST_DIM)
        x = data[('data', 'BANDS')].astype(np.float32)
        mask = data[('mask', 'CLM')] == 1
        try:
            ref_mask = data[('mask_timeless', 'NOT_DECLARED')].squeeze()
        except KeyError as error:
            print(error)
            print('No mask for the image')
            ref_mask = np.zeros(shape=self.DEST_DIM)
        if str(sample_path).startswith('h_'):
            x = x[:, :, ::-1, :]
            y = y[:, ::-1]
        if str(sample_path).startswith('v_'):
            x = x[:, ::-1, :, :]
            y = y[::-1, :]
        # Reduce the temporary features via adaptive pooling
        # and upscale the image with cubic interpolation:
        if self.model_name == 'gcnn' or self.model_name == 'unet':
            x = np.rollaxis(x, 3, 0)
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3]).T
            x = adaptive_max_pool1d(
                torch.from_numpy(x),
                output_size=self.time_step_max) \
                .numpy().T.reshape(self.time_step_max, *self.INIT_DIM)
            x_upscale = np.zeros(shape=(x.shape[0], *self.DEST_DIM))
            for idx in range(x.shape[0]):
                x_upscale[idx] = cv2.resize(
                    x[idx], dsize=self.DEST_DIM,
                    interpolation=cv2.INTER_CUBIC)
        else:
            x_upscale = x
        return x_upscale.astype(np.float32), y.astype(int), mask, ref_mask

    def upscale(self, x: torch.Tensor) -> np.ndarray:
        x = torch.moveaxis(x, -1, 1)
        x = F.interpolate(x, scale_factor=self.SCALE,
                          mode='bicubic', align_corners=False)
        x = torch.moveaxis(x, 1, -1)
        return x.numpy()

    def __getitem__(self, index: int) -> \
            Union[Tuple[np.ndarray, np.ndarray], List[Data]]:
        x, y, mask, ref_mask = self.load_and_upscale_img(index)
        x = (x - x.mean()) / x.std()
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        patches = []
        for i, j in product(range(0, x.shape[1], self.patch_size),
                            range(0, x.shape[2], self.patch_size)):
            x_patch = x[:, i:i + self.patch_size, j:j + self.patch_size]
            if self.model_name == 'gcnn':
                x_patch = x_patch.reshape(
                    x_patch.shape[0],
                    x_patch.shape[1] * x_patch.shape[2]).T
                y_patch = torch.unsqueeze(
                    y[i:i + self.patch_size,
                      j:j + self.patch_size].reshape(-1), -1)
                ref_mask_patch = ref_mask[i:i + self.patch_size,
                                          j:j + self.patch_size].ravel()
                ref_mask_patch = torch.tensor(ref_mask_patch, dtype=torch.long)
                patches.append((Data(x=x_patch, y=y_patch,
                                     edge_index=self.adj_matrix,
                                     x_mask=ref_mask_patch)))

            elif self.model_name == 'unet':
                ref_mask_patch = ref_mask[i:i + self.patch_size,
                                          j:j + self.patch_size].ravel()
                ref_mask_patch = torch.tensor(ref_mask_patch, dtype=torch.long)
                patches.append((x_patch, ref_mask_patch,
                                y[i:i + self.patch_size,
                                  j:j + self.patch_size]))

            elif self.model_name == 'rf':
                x_patch = self.upscale(x_patch)
                mask_patch = mask[:, i:i + self.patch_size,
                                  j:j + self.patch_size]
                ref_mask_patch = \
                    ref_mask[i*self.SCALE:i*self.SCALE +
                             self.patch_size*self.SCALE,
                             j*self.SCALE:j*self.SCALE +
                             self.patch_size*self.SCALE]
                mask_patch = self.upscale(mask_patch) > 0.5
                patches.append((
                    get_features(
                        x_patch, mask_patch, self.features_list,
                        kernel_size=self.kernel_size, device=self.device),
                    ref_mask_patch,
                    y[i*self.SCALE:i*self.SCALE+self.patch_size*self.SCALE,
                      j*self.SCALE:j*self.SCALE+self.patch_size*self.SCALE]))
            else:
                raise ArgumentError(
                    'The value of the "model_name" argument is incorrect.')
        if self.model_name != 'rf':
            return patches
        # If the utilized model is random forest:
        # If m == True, then ignore the calculation of the metrics:
        x, m, y = zip(*patches)
        x = np.stack(x, axis=0).astype(np.float32)
        x = x.reshape(-1, x.shape[-1])
        m = np.stack(m, axis=0).astype(bool).reshape(-1)
        y = np.stack(y, axis=0).astype(bool).reshape(-1)
        return x, m, y
