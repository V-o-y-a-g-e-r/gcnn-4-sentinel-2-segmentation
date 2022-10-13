import json
import os
from typing import Dict, List

import joblib
import mlflow
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support)
from torch.utils.data import Dataset

LOGGING_EXCLUDED_PARAMS = ['run_name', 'experiment_name', 'use_mlflow',
                           'verbose']
MEAN = 0


def rf_loop(model: RandomForestClassifier, generator: Dataset,
            dest_path: str, mode: str, metrics: Dict[str, float]):
    if mode == 'train':
        train_mode_flag = 1
    elif mode == 'val' or mode == 'test':
        train_mode_flag = 0
    else:
        raise ValueError(f'Mode set to incorrect value: '
                         f'"{mode}", should be "train", "val" or "test"')
    for img_batch in generator:
        y_true = img_batch[2]
        mask = img_batch[1]
        if train_mode_flag:
            model = model.fit(img_batch[0], y_true)
            joblib.dump(model,
                        os.path.join(dest_path, 'model.joblib.pkl'),
                        compress=9)
        model = joblib.load(os.path.join(dest_path, 'model.joblib.pkl'))
        y_pred = model.predict(img_batch[0])
        loss = log_loss(y_true=y_true, y_pred=y_pred)
        loss /= img_batch[0].shape[0]
        metrics[f'{mode}_loss'].append(loss)
        y_pred = (y_pred > 0.5).astype(bool)
        calculate_metrics(y_true, y_pred, metrics, mode)
        calculate_metrics(y_true, y_pred, metrics, mode, mask)


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      metrics: Dict[str, float],
                      mode: str, mask: np.ndarray = None) -> None:
    if mask is None:
        mask_flag = ''
    else:
        mask_flag = 'masked_'
        y_true = y_true[~mask]
        y_pred = y_pred[~mask]
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_pred=y_pred, y_true=y_true, average='binary')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    metrics[f'{mask_flag}{mode}_acc'].append(acc)
    metrics[f'{mask_flag}{mode}_precision'].append(precision)
    metrics[f'{mask_flag}{mode}_recall'].append(recall)
    metrics[f'{mask_flag}{mode}_f_score'].append(f_score)
    metrics[f'{mask_flag}{mode}_mcc'].append(mcc)


def log_dict_as_str_to_mlflow(dict_as_string: str) -> None:
    """
    Log a string which represents a dictionary to MLflow
    :param dict_as_string: A string with dictionary format
    :return: None
    """
    try:
        to_log = json.loads(dict_as_string)
    except json.decoder.JSONDecodeError:
        to_log = yaml.load(dict_as_string)
    mlflow.log_params(to_log)


def log_params_to_mlflow(args: Dict) -> None:
    """
    Log provided arguments as dictionary to mlflow.
    :param args: Arguments to log
    """
    args['artifacts_storage'] = args.pop('dest_path')
    for arg in args.keys():
        if arg not in LOGGING_EXCLUDED_PARAMS and args[arg] is not None:
            if type(args[arg]) is list:
                args[arg] = ''.join(args[arg])
                if args[arg] == "":
                    continue
            elif arg == 'noise_params':
                log_dict_as_str_to_mlflow(args[arg])
                continue
            mlflow.log_param(arg, args[arg])


def log_metrics_to_mlflow(metrics: Dict[str, float],
                          fair: bool = False) -> None:
    """
    Log provided metrics to mlflow
    :param metrics: Metrics in a dictionary
    :param fair: Whether to add '_fair' suffix to the metrics name
    :return: None
    """
    for metric in metrics.keys():
        if metric != 'Stats':
            if fair:
                mlflow.log_metric(metric + '_fair', metrics[metric][MEAN])
            else:
                mlflow.log_metric(metric, metrics[metric][MEAN])


def get_metrics_dict() -> Dict[str, List[float]]:
    base = {'loss': [], 'acc': [], 'precision': [],
            'recall': [], 'f_score': [], 'mcc': [], 'dice': []}
    metrics = {}
    for mode in ['train', 'val', 'test']:
        for key in base.keys():
            metrics[f'{mode}_{key}'] = []
    for key in list(metrics.keys()):
        metrics[f'masked_{key}'] = []
    return metrics
