import unittest

import torch

from experimenting.models.metrics import AUC, MPJPE, PCK
from experimenting.utils import average_loss


def get_random_mask(batch_size, n_joints):
    return (torch.FloatTensor(batch_size, n_joints).uniform_() > 0.8).type(
        torch.bool)


class TestAverageLoss(unittest.TestCase):
    def test_call(self):
        input_value = torch.randn(10, 10)
        mask = get_random_mask(10, 10)
        try:
            result = average_loss(input_value, mask)
            self.assertIsNotNone(result)
        except Exception:
            self.fail("average_loss raised exception!")


class TestBaseMetric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is TestBaseMetric:
            raise unittest.SkipTest("Skip TestBaseMetric tests")

        super(TestBaseMetric, cls).setUpClass()

    def setUp(self):

        self.metric = None

    def test_init(self):
        self.assertIsNotNone(self.metric)

    def test_accuracy_vanilla(self):
        gt = torch.randn(self.batch_size, self.n_joints, 3)

        result = self.metric(gt, gt)

        self.assertEqual(result.sum(), 0)

    def test_accuracy_no_mask(self):
        gt = torch.randn(self.batch_size, self.n_joints, 3)

        pred = gt - torch.randn(self.batch_size, self.n_joints, 3)

        result = self.metric(pred, gt)

        self.assertGreater(result.sum(), 0)

    def test_accuracy_with_mask(self):
        gt = torch.randn(self.batch_size, self.n_joints, 3)

        mask = get_random_mask(self.batch_size, self.n_joints)
        pred = gt - torch.randn(self.batch_size, self.n_joints, 3)

        result_mask = self.metric(pred, gt, mask)

        self.assertGreater(result_mask.sum(), 0)


class TestMPJPE(TestBaseMetric):
    def setUp(self):
        self.batch_size = 32
        self.n_joints = 13

        self.metric = MPJPE(reduction=None)


class TestPCK(TestBaseMetric):
    def setUp(self):
        self.batch_size = 32
        self.n_joints = 13

        self.metric = PCK(reduction=average_loss)

    def test_accuracy_vanilla(self):
        gt = torch.randn(self.batch_size, self.n_joints, 3)

        result = self.metric(gt, gt)

        self.assertEqual(result.sum(), 1.)  # If every joint match
        # -> sum of results equal number of joint predictions


class TestAUC_reduced(TestBaseMetric):
    def setUp(self):
        self.batch_size = 32
        self.n_joints = 13

        self.metric = AUC(reduction=average_loss,
                          auc_reduction=torch.mean,
                          start_at=1,
                          end_at=500,
                          step=40)

    def test_accuracy_vanilla(self):
        gt = torch.randn(self.batch_size, self.n_joints, 3)

        result = self.metric(gt, gt)

        self.assertEqual(result.sum(), 1.)
        # -> starting thresholds at 1 lead to a maximum of 1.>


class TestAUC_raw(TestBaseMetric):
    def setUp(self):
        self.batch_size = 32
        self.n_joints = 13

        self.metric = AUC(reduction=average_loss, auc_reduction=None)

    def test_accuracy_vanilla(self):
        gt = torch.randn(self.batch_size, self.n_joints, 3)

        result = self.metric(gt, gt)

        self.assertAlmostEqual(result.sum(), len(result) - 1)
        # -> sum of results equal number of joint predictions


if __name__ == '__main__':
    unittest.main()
