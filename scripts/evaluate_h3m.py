import json
import logging
import os

import hydra
from omegaconf import DictConfig

from experimenting import agents
from experimenting.utils.skeleton_helpers import Skeleton
from experimenting.utils.trainer import HydraTrainer

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train/eval.yaml')
def main(cfg: DictConfig) -> None:
    trainer = HydraTrainer(cfg)
    normalized_predictions = trainer.get_raw_test_outputs()
    torso_length = 400
    for idx, pred in enumerate(normalized_predictions):
        sk = Skeleton(pred)
        test_index = trainer.core.test_indexes[idx]
        intr_matrix, extr_matrix = self.get_matrices_from_id(test_index)
        timestamp = self.get_timestamp_from_id(test_index)
        sk.denormalize(
            trainer.core.in_shape[0],
            trainer.core.in_shape[1],
            intr_matrix,
            torso_length=trainer.core.avg_torso_length,
        )


if __name__ == '__main__':
    main()
