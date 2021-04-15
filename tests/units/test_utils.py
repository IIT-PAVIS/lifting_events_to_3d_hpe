import unittest
from unittest import mock

import torch

from experimenting import utils


class TestUtilites(unittest.TestCase):
    @mock.patch("os.path.exists")
    @mock.patch("hydra.utils.instantiate")
    @mock.patch("experimenting.utils.trainer.instantiate_new_model")
    @mock.patch("experimenting.utils.trainer.get_training_params")
    @mock.patch("experimenting.dataset.DataModule")
    def test_trainer_init(
        self,
        mocked_datamodule,
        mocked_new_model,
        training_params,
        mocked_instantiate,
        mocked_os_exists,
    ):
        mocked_os_exists.return_value = False
        cfg = mock.MagicMock()
        trainer = utils.trainer.HydraTrainer(cfg)
        mocked_datamodule.assert_called_once()
        mocked_new_model.assert_called_once()
        mocked_instantiate.assert_called_once()

    @mock.patch("os.path.exists")
    @mock.patch("hydra.utils.instantiate")
    @mock.patch("experimenting.utils.trainer.instantiate_new_model")
    @mock.patch("experimenting.utils.trainer.get_training_params")
    @mock.patch("experimenting.dataset.DataModule")
    @mock.patch("pytorch_lightning.Trainer.fit")
    def test_trainer_fit(
        self,
        mocked_fit,
        mocked_datamodule,
        mocked_new_model,
        training_params,
        mocked_instantiate,
        mocked_os_exists,
    ):
        mocked_os_exists.return_value = False
        cfg = mock.MagicMock()
        trainer = utils.trainer.HydraTrainer(cfg)

        trainer.fit()

        mocked_fit.assert_called_once()

    @mock.patch("os.path.exists")
    @mock.patch("hydra.utils.instantiate")
    @mock.patch("experimenting.utils.trainer.instantiate_new_model")
    @mock.patch("experimenting.utils.trainer.get_training_params")
    @mock.patch("experimenting.dataset.DataModule")
    @mock.patch("pytorch_lightning.Trainer.test")
    def test_trainer_test(
        self,
        mocked_test,
        mocked_datamodule,
        mocked_new_model,
        training_params,
        mocked_instantiate,
        mocked_os_exists,
    ):
        mocked_os_exists.return_value = False
        cfg = mock.MagicMock()
        trainer = utils.trainer.HydraTrainer(cfg)

        trainer.test(save_results=False)

        mocked_test.assert_called_once()
