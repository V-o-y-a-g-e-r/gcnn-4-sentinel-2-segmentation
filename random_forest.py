from enum import Enum
from itertools import product
from typing import List

import numpy as np
import torch
from torch.nn import functional as F


class Features(Enum):
    MEAN = 0
    STD = 1
    MEDIAN = 2
    QUANTILE = 3
    MIN = 4
    MAX = 5
    RANGE = 6


def calculate_features(X: torch.Tensor, mask: torch.Tensor,
                       features: List[Features]) -> List[torch.Tensor]:
    if not isinstance(features, list):
        features = [features]
    results = []
    Num = torch.sum(~torch.isnan(X), dim=0)
    mean = torch.nansum(X, dim=0)/Num

    if Features.MEAN in features:
        results.append(mean)

    if Features.STD in features:
        results.append(nan_std(X, Num, mean))
    del Num

    if Features.MEDIAN in features:
        results.append(torch.nanmedian(X, dim=0).values)

    if Features.QUANTILE in features:
        results = results + \
            [*torch.nanquantile(
                X, torch.Tensor([0.25, 0.75])
                .to(X.device), dim=0)]

    X[mask] = mean.unsqueeze(0).repeat_interleave(
        X.shape[0], dim=0)[mask]
    del mean
    if Features.MIN in features:
        results.append(X.min(0).values)

    if Features.MAX in features:
        results.append(X.max(0).values)

    if Features.MIN in features:
        results.append(X.max(0).values - X.min(0).values)
    return results


def get_features(X: np.ndarray, mask: np.ndarray,
                 features: List[Features],
                 kernel_size: int, device: 'str') -> torch.Tensor:
    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size]
    result = []
    mask = torch.from_numpy(mask).repeat_interleave(
        X.shape[-1], dim=-1).to(device)
    X = torch.from_numpy(X)
    # X: (N, W, H, 12)
    for i, k in enumerate(kernel_size):
        X = X.to(device)
        x = expand_neighbors_dim(X, k)
        if i < (len(kernel_size)-1):
            X = X.cpu()
        else:
            del X
        # X: (k*k*N, W, H, 12)
        temp_mask = expand_neighbors_dim(
            mask.to(torch.float), k).to(torch.bool)
        x[temp_mask] = float('NaN')
        result += calculate_features(x, temp_mask, features)
    del mask
    result = torch.cat(result, dim=-1).detach().cpu().numpy()
    return result


def nan_std(X: torch.Tensor,
            N_unmasked: torch.Tensor,
            mean: torch.Tensor):
    mean = torch.repeat_interleave(
        mean.unsqueeze(0), X.shape[0], dim=0)
    result = torch.sqrt(torch.nansum(
        torch.pow(X - mean, 2), dim=0)/N_unmasked)
    return result


def expand_neighbors_dim(x: torch.Tensor,
                         kernel_size: int) -> torch.Tensor:
    device = x.device
    kernel_size = kernel_size//2
    kernel_indices = list(
        product(range(-kernel_size, kernel_size+1),
                range(-kernel_size, kernel_size+1)))
    kernel_indices.remove((0., 0.))
    x = torch.moveaxis(x, -1, 1)
    output_shape = x.shape
    x = x.unsqueeze(1)
    for i, j in kernel_indices:
        theta = torch.Tensor([[1, 0, 2*i/x.shape[1]],
                              [0, 1, 2*j/x.shape[1]]])\
            .to(device).unsqueeze(0)\
            .repeat_interleave(x.shape[0], dim=0)
        grid = F.affine_grid(theta, output_shape)
        x = torch.cat([
            x, F.grid_sample(x[:, 0, ...], grid,
                             mode='bicubic',
                             padding_mode='border')
            .unsqueeze(1)], dim=1)
    x = torch.moveaxis(x, 2, -1)
    x = x.reshape((-1, *x.shape[2:]))
    return x
