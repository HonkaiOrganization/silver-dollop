import pandas as pd
import numpy as np
import os
import glob


def load_and_normalize_csv(csv_path):
    """
    读取 CSV 并提取 (x, y) 坐标，进行空间归一化。
    返回: (Frames, 17, 2) 的 numpy 数组
    """
    df = pd.read_csv(csv_path)
    # 剔除 frame_id, person_id，提取 17 个关节的 x, y, conf
    raw_data = df.iloc[:, 2:].values
    coords = raw_data.reshape(-1, 17, 3)[:, :, :2] # 仅保留 x, y

    # 空间归一化
    # 1. 中心化：以左右髋关节 (索引 11, 12) 的中点为原点
    mid_hip = (coords[:, 11, :] + coords[:, 12, :]) / 2.0
    coords = coords - mid_hip[:, np.newaxis, :]

    # 2. 尺度缩放：以双肩 (索引 5, 6) 距离为基准
    shoulder_vec = coords[:, 6, :] - coords[:, 5, :]
    shoulder_width = np.linalg.norm(shoulder_vec, axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-6) # 防止除零
    coords = coords / shoulder_width[:, np.newaxis, :]

    return coords


def load_csv_directory(dir_path, label):
    """
    加载目录下所有 CSV 文件，返回 (data_segments, labels) 元组。
    data_segments: list of numpy arrays, 每个 array shape 为 (Frames, 17, 2)
    labels: list of int, 对应每个 segment 的标签
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
    根据配置文件加载数据集。
    
    Args:
        data_config: 配置文件中的 data 部分
        subset: 'train' 或 'test'
    
    Returns:
        (data_segments, labels) 元组
    """
    subset_config = data_config.get('subsets', {}).get(subset, {})
    
    all_segments = []
    all_labels = []
    
    # 加载正常数据 (label=1)
    normal_dir = subset_config.get('normal')
    if normal_dir and os.path.isdir(normal_dir):
        segments, labels = load_csv_directory(normal_dir, label=1)
        all_segments.extend(segments)
        all_labels.extend(labels)
    
    # 加载不规范数据 (label=0)
    abnormal_dir = subset_config.get('abnormal')
    if abnormal_dir and os.path.isdir(abnormal_dir):
        segments, labels = load_csv_directory(abnormal_dir, label=0)
        all_segments.extend(segments)
        all_labels.extend(labels)
    
    return all_segments, all_labels