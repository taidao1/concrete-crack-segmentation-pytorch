"""固定种子的数据划分与 DataLoader 构建。"""

import math
import random

from torch.utils.data import DataLoader

from .data import image_datasets, image_list


def split_list(sample_ids, ratios=[0.7,0.15,0.15], seed=1):
    '''这是一个将一个列表进行打乱按比例分类的函数，有种子保证可复现性'''
    data = sample_ids.copy()
    rng = random.Random(seed)
    rng.shuffle(data)
    n = len(data)
    if len(ratios) != 3:
        raise ValueError("比例长度必须为3")
    if any(r < 0 for r in ratios):
        raise ValueError("每个比例都要大于0")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("比例之和必须为1")

    sizes = [int(r*n) for r in ratios]
    sizes[0] += n - sum(sizes)

    result , start = [], 0
    for s in sizes:
        result.append(data[start:start+s])
        start += s
    train_ids, val_ids, test_ids = result[0], result[1], result[2]
    return train_ids, val_ids, test_ids


def makedatasets(rgb_path, BW_path,ratios=[0.7,0.15,0.15], seed=1, rand_crop = True):
    '''这个一个集成的函数，把前面dataset类和打乱结合在一起，最终输出三个dataset'''
    all_ids = image_list(rgb_path, BW_path)
    train_ids, val_ids, test_ids = split_list(all_ids,ratios,seed)
    train_dataset = image_datasets(rgb_path, BW_path, train_ids, rand_crop)
    val_dataset = image_datasets(rgb_path, BW_path, val_ids)
    test_dataset = image_datasets(rgb_path, BW_path, test_ids)
    return train_dataset, val_dataset, test_dataset


def makedataloader(dataset,shuffle=True):
    loader = DataLoader(dataset, batch_size = 2, shuffle=shuffle)
    return loader
