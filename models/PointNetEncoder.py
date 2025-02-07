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
        
        self.conv1 = nn.Conv1d(3, 32, 1)  # Reduced from 64
        self.conv2 = nn.Conv1d(32, 64, 1)  # Reduced from 128
        self.conv3 = nn.Conv1d(64, 256, 1)  # Reduced from 1024
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)

        if self.feature_transform:
            self.tnet2 = TNet(k=64)

    def forward(self, x):
        batch_size, _, num_points = x.shape
        trans = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.tnet2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)

        return x, trans_feat
