import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 8, 1)  # Reduced from 16
        self.conv2 = nn.Conv1d(8, 16, 1)  # Reduced from 32
        self.conv3 = nn.Conv1d(16, 32, 1)  # Reduced from 128
        self.fc1 = nn.Linear(32, 32)      # Reduced from 64
        self.fc2 = nn.Linear(32, 16)      # Reduced from 32
        self.fc3 = nn.Linear(16, k * k)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(16)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(batch_size, self.k, self.k)
        identity = torch.eye(self.k, device=x.device).view(1, self.k, self.k)
        return x + identity
