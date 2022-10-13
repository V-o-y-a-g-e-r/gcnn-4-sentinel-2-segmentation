import os
from ctypes import ArgumentError
from time import time
from typing import Dict, List, Union

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import Dataset

from gcnns import utils


def dice_loss1(y_pred: torch.Tensor,
               y_true: torch.Tensor,
               smooth: float = 1.,
               p: float = 2.) -> torch.Tensor:
    """
    Implementation of the dice loss from the following publication:
    @inproceedings{milletari2016v,
    title={V-net: Fully convolutional neural networks
    for volumetric medical image segmentation},
    author={Milletari, Fausto and Navab, Nassir and Ahmadi, Seyed-Ahmad},
    booktitle={2016 fourth international conference on 3D vision (3DV)},
    pages={565--571},
    year={2016},
    organization={IEEE}
    }
    """
    assert y_pred.shape[0] == y_true.shape[0], \
        'Predict and target batch size do not match.'
    assert y_pred.ndim == 2 and y_true.ndim == 2
    num = torch.sum(torch.mul(y_pred, y_true), dim=1) + smooth
    den = torch.sum(y_pred.pow(p) + y_true.pow(p), dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def dice_loss2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    smooth = 1.
    iflat = y_pred.view(-1)
    tflat = y_true.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def deep_model_loop(model, model_name: str, generator: Dataset,
                    optimizer: torch.optim.Optimizer, mode: str,
                    epoch_metrics: Dict, device: str) -> None:
    if mode == 'train':
        train_mode_flag = 1
        model.train()
    elif mode == 'val' or mode == 'test':
        train_mode_flag = 0
        model.eval()
    else:
        raise ValueError(f'Mode set to incorrect value: '
                         f'"{mode}", should be "train", "val" or "test"')
    for img_batch in generator:
        img_batch_loss, img_batch_dice = 0, 0
        img_batch_y_pred, img_batch_y_true, img_masks = [], [], []
        for patch_batch in img_batch:

            if train_mode_flag:
                optimizer.zero_grad()

            if model_name == 'gcnn':
                y_pred = model(patch_batch.x.to(device),
                               patch_batch.edge_index.to(device))
                mask = patch_batch.x_mask.numpy().astype(bool)
                y_true = patch_batch.y.to(device)

            elif model_name == 'unet':
                y_pred = model(patch_batch[0].to(device)).squeeze()
                mask = patch_batch[1].numpy().reshape(-1).astype(bool)
                y_true = patch_batch[2].to(device)
            else:
                raise ArgumentError(
                    'The value of the "model_name" argument is incorrect.')
            y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
            y_true = y_true.contiguous().view(y_true.shape[0], -1)
            loss = binary_cross_entropy(input=y_pred, target=y_true)
            if model_name == 'gcnn':
                dice = torch.tensor([0]).to(device)
            elif model_name == 'unet':
                dice = dice_loss1(y_pred=y_pred, y_true=y_true)
            else:
                raise ArgumentError(
                    'The value of the "model_name" argument is incorrect.')
            if train_mode_flag:
                total_loss = loss + dice
                total_loss.backward()
                optimizer.step()
            img_batch_loss += float(loss.detach().cpu().numpy())
            img_batch_dice += float(dice.detach().cpu().numpy())
            y_pred = (y_pred.detach().cpu().numpy().squeeze()
                      > 0.5).astype(bool).reshape(-1)
            y_true = y_true.cpu().numpy().reshape(-1)
            img_batch_y_pred.append(y_pred)
            img_batch_y_true.append(y_true)
            img_masks.append(mask)
        img_batch_y_pred = np.concatenate(img_batch_y_pred)
        img_batch_y_true = np.concatenate(img_batch_y_true)
        img_masks = np.concatenate(img_masks)
        utils.calculate_metrics(
            img_batch_y_true, img_batch_y_pred, epoch_metrics, mode)
        utils.calculate_metrics(
            img_batch_y_true, img_batch_y_pred, epoch_metrics, mode, img_masks)
        img_batch_loss /= len(img_batch)
        img_batch_dice /= len(img_batch)
        epoch_metrics[f'{mode}_loss'].append(img_batch_loss)
        epoch_metrics[f'{mode}_dice'].append(img_batch_dice)


def train_deep_models(model, model_name: str,
                      train_generator: Dataset,
                      val_generator: Dataset,
                      test_generator: Union[Dataset, List],
                      epochs: int, learning_rate: float, verbose: int,
                      dest_path: str, metrics: Dict[str, float],
                      patience: int, load_checkpoint: bool,
                      device: str = torch.device('cuda')):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    timeout = time() + 86400
    for epoch in range(epochs):
        if time() > timeout:
            print('Hitting the time limit, stopping training...')
            break
        epoch_metrics = utils.get_metrics_dict()
        # Train the model:
        deep_model_loop(model, model_name, train_generator, optimizer,
                        'train', epoch_metrics, device)
        # Evaluate the model:
        deep_model_loop(model, model_name, val_generator, optimizer,
                        'val', epoch_metrics, device)
        # After the epoch ends append its metrics:
        print(u'\u2500' * 60)
        print(f'Epoch number: {epoch+1:5}/{epochs}')
        for key in metrics.keys():
            if 'test' not in key.lower():
                val = np.mean(epoch_metrics[key])
                metrics[key].append(val)
                print(f'Epoch mean of\t{key.upper()}:\t{val:15.3f}')
        # Save the model if the loss minimized:
        if epoch == 0 or metrics['val_loss'][-1] < \
                min(metrics['val_loss'][:-1]):
            print('Val loss improvement, thus saving model')
            torch.save(model.state_dict(), os.path.join(dest_path, 'model'))
        # Load the best state of the model:
        if load_checkpoint:
            if epoch > 0 and metrics['val_loss'][-1] > \
                    min(metrics['val_loss'][:-1]):
                print('Loading the best saved model')
                model.load_state_dict(torch.load(
                    os.path.join(dest_path, 'model')))
        # Check whether to bail from training
        # because of no improvement in val loss:
        if epoch > patience:
            if min(metrics['val_loss'][-patience:]) > \
                    min(metrics['val_loss'][:-patience]):
                print('Stopping training, no improvement',
                      f'for {patience} epochs')
                break
    # Load the best state of the model:
    if len(test_generator) > 0:
        model.load_state_dict(torch.load(os.path.join(dest_path, 'model')))
        deep_model_loop(model, model_name, test_generator, optimizer,
                        'test', metrics, device)
        for key in metrics.keys():
            if 'test' in key.lower():
                val = np.mean(metrics[key])
                metrics[key] = val
                print(f'Epoch mean of\t{key.upper()}:\t{val:15.3f}')
