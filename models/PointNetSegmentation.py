import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNetEncoder import PointNetEncoder

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes, feature_transform=False):
        super(PointNetSegmentation, self).__init__()
        self.k = num_classes
        self.feat = PointNetEncoder(global_feat=False, feature_transform=False)
        
        self.conv1 = nn.Conv1d(288, 128, 1)  # 256 (global) + 32 (local) = 288
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        batch_size, _, num_points = x.shape
        local_features, trans_feat = self.feat(x)
        global_features = torch.max(local_features, 2, keepdim=True)[0]
        global_features = global_features.repeat(1, 1, num_points)

        x = torch.cat([local_features, global_features], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        return F.log_softmax(x, dim=1), trans_feat
