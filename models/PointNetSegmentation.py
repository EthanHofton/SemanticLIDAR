import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNetEncoder import PointNetEncoder

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes, feature_transform=False):
        super(PointNetSegmentation, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=False, feature_transform=feature_transform)  # Keep per-point features
        
        # MLP layers for per-point classification
        self.conv1 = nn.Conv1d(1088, 512, 1)  # 1024 (global) + 64 (local) = 1088
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size, _, num_points = x.shape
        
        # Extract features
        local_features, trans_feat = self.feat(x)  # (B, 64, N)
        global_features = torch.max(local_features, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_features = global_features.repeat(1, 1, num_points)  # (B, 1024, N)

        # Concatenate local and global features
        x = torch.cat([local_features, global_features], dim=1)  # (B, 1088, N)

        # Per-point classification
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # (B, num_classes, N)

        return F.log_softmax(x, dim=1), trans_feat  # Log-softmax for NLL loss
