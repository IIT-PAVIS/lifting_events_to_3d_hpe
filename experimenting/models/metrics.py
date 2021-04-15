"""
Metrics implementation for 3D human pose comparisons
"""

import torch
from torch import nn

__all__ = ['MPJPE', 'AUC', 'PCK']


class BaseMetric(nn.Module):
    def forward(self, y_pr, points_gt, gt_mask=None):
        """
        Base forward method for metric evaluation
        Args:
            y_pr: 3D prediction of joints, tensor of shape (BATCH_SIZExN_JOINTSx3)
            points_gt: 3D gt of joints, tensor of shape (BATCH_SIZExN_JOINTSx3)
            gt_mask: boolean mask, tensor of shape (BATCH_SIZExN_JOINTS). 
            Applied to results, if provided

        Returns:
            Metric as single value, if reduction is given, or as a tensor of values
        """
        pass


class MPJPE(BaseMetric):
    def __init__(self, reduction=None, confidence=0, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        if gt_mask is not None:
            points_gt[~gt_mask] = 0

        dist_2d = torch.norm((points_gt - y_pr), dim=-1)

        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d


class PCK(BaseMetric):
    """
    Percentage of correct keypoints according to a thresold value. Usually
    default threshold is 150mm
    """
    def __init__(self, reduction=None, threshold=150, **kwargs):
        super().__init__(**kwargs)
        self.thr = threshold
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        if gt_mask is not None:
            points_gt[~gt_mask] = 0

        dist_2d = (torch.norm((points_gt - y_pr), dim=-1) < self.thr).double()
        if self.reduction:
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d


class AUC(BaseMetric):
    """
    Area Under the Curve for PCK metric, 
    at different thresholds (from 0 to 800)
    """
    def __init__(self,
                 reduction=None,
                 auc_reduction=torch.mean,
                 start_at=0,
                 end_at=500,
                 step=30,
                 **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.auc_reduction = auc_reduction
        self.thresholds = torch.linspace(start_at, end_at, step).tolist()

    def forward(self, y_pr, points_gt, gt_mask=None):
        pck_values = torch.DoubleTensor(len(self.thresholds))
        for i, threshold in enumerate(self.thresholds):
            _pck = PCK(self.reduction, threshold=threshold)
            pck_values[i] = _pck(y_pr, points_gt, gt_mask)

        if self.auc_reduction:
            pck_values = torch.mean(pck_values)

        return pck_values
