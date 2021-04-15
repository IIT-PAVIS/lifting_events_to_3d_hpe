import unittest
from unittest import mock

import pytorch_lightning as pl
import torch
from torch import nn

from experimenting.models.dhp19 import DHP19Model


class TestDHP19Model(unittest.TestCase):
    def setUp(self):
        self.n_channels = 1
        self.n_joints = 13
        self.model = DHP19Model(self.n_channels, self.n_joints)

    def test_init(self):
        self.assertIsInstance(self.model, nn.Module)

    def test_forward(self):
        b_x = torch.rand((1, self.n_channels, 224, 224))
        output = self.model(b_x)

        self.assertEqual(output.shape, (1, self.n_joints, 224, 224))
