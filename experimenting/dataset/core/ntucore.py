import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import io

from ..utils import get_file_paths
from .base import BaseCore


class NTUCore(BaseCore):
    DEFAULT_TEST_SUBJECTS = [18, 19, 20]

    def __init__(
        self, name, data_dir, labels_dir, test_subjects, partition, *args, **kwargs,
    ):
        super(NTUCore, self).__init__(name, partition)
        self.file_paths = NTUCore._get_file_paths(data_dir)

        if test_subjects is None:
            self.subjects = NTUCore.DEFAULT_TEST_SUBJECTS
        else:
            self.subjects = test_subjects

    @staticmethod
    def load_frame(path):
        x = np.load(path, allow_pickle=True) / 255.0
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        return x

    def get_frame_from_id(self, idx):
        img_name = self.file_paths[idx]
        x = self.load_frame(img_name)
        return x

    @staticmethod
    def get_frame_info(path):

        dir_name = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(path)))
        )
        info = {"subject": int(dir_name[-2:])}
        return info

    def get_test_subjects(self):
        return self.subjects

    @staticmethod
    def _get_file_paths(data_dir):
        file_paths = []
        for root, dirs, files in os.walk(data_dir):
            if "part_" in root:
                for f in files:
                    file_path = os.path.join(root, f)
                    file_paths.append(file_path)
        return file_paths


def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes
