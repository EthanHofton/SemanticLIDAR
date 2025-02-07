import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TNet import TNet

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        self.tnet1 = TNet(k=3)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.tnet2 = TNet(k=64)

    def forward(self, x):
        batch_size, _, num_points = x.shape
        
        # Apply input transformation
        trans = self.tnet1(x)
        x = x.transpose(2, 1)  # Swap to (B, N, 3)
        x = torch.bmm(x, trans)  # Apply learned transformation
        x = x.transpose(2, 1)  # Swap back to (B, 3, N)
        
        # Per-point feature extraction
        x = F.relu(self.bn1(self.conv1(x)))

        # Apply feature transformation if enabled
        if self.feature_transform:
            trans_feat = self.tnet2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)  # (B, 1024)

        return x, trans_feat

