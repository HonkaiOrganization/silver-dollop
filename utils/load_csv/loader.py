import os
import glob
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_and_normalize_csv(csv_path):
    """
    Read CSV and extract (x, y) coordinates with spatial normalization.
    Returns: numpy array of shape (Frames, 17, 2)
    """
    df = pd.read_csv(csv_path)
    raw_data = df.iloc[:, 2:].values
    coords = raw_data.reshape(-1, 17, 3)[:, :, :2]

    mid_hip = (coords[:, 11, :] + coords[:, 12, :]) / 2.0
    coords = coords - mid_hip[:, np.newaxis, :]

    shoulder_vec = coords[:, 6, :] - coords[:, 5, :]
    shoulder_width = np.linalg.norm(shoulder_vec, axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    coords = coords / shoulder_width[:, np.newaxis, :]

    return coords


def load_csv_directory(dir_path, label):
    """
    Load all CSV files from a directory, returning (data_segments, labels) tuple.
    """
    csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
    data_segments = []
    labels = []

    for csv_file in sorted(csv_files):
        coords = load_and_normalize_csv(csv_file)
        data_segments.append(coords)
        labels.append(label)

    return data_segments, labels


def load_dataset_from_config(data_config, subset='train'):
    """
    Load dataset from configuration.

    Args:
        data_config: Data section from config file
        subset: 'train' or 'test'

    Returns:
        (data_segments, labels) tuple
    """
    subset_config = data_config.get('subsets', {}).get(subset, {})

    all_segments = []
    all_labels = []

    normal_dir = subset_config.get('normal')
    if normal_dir and os.path.isdir(normal_dir):
        segments, labels = load_csv_directory(normal_dir, label=1)
        all_segments.extend(segments)
        all_labels.extend(labels)

    abnormal_dir = subset_config.get('abnormal')
    if abnormal_dir and os.path.isdir(abnormal_dir):
        segments, labels = load_csv_directory(abnormal_dir, label=0)
        all_segments.extend(segments)
        all_labels.extend(labels)

    return all_segments, all_labels
