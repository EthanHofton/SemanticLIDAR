import torch.nn.functional as F

def pointnet_segmentation_loss(pred, target):
    """
    Computes segmentation loss.
    pred: (B, num_classes, N)  - Log probabilities
    target: (B, N)             - Ground truth labels
    """
    return F.nll_loss(pred, target)
