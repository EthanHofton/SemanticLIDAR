import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.utils import to_dense_batch
from torch_cluster import knn_graph

class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, 64),
                nn.ReLU(),
                nn.Linear(64, out_channels)
            ), 
            k=k  # Number of nearest neighbors
        )

    def forward(self, x, batch):
        return self.conv(x, batch)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, num_points, channels = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into q, k, v
        q, k, v = map(lambda t: t.view(batch_size, num_points, self.heads, -1).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_points, -1)
        return self.fc_out(out)

class get_model(nn.Module):
    def __init__(self, num_classes, k=20, heads=4):
        super().__init__()

        # Local Feature Extraction with EdgeConv
        self.edgeconv1 = EdgeConvBlock(3, 64, k=k)
        self.edgeconv2 = EdgeConvBlock(64, 128, k=k)

        # Self-Attention Layer (Global Feature Enhancement)
        self.attention = SelfAttention(128, heads=heads)

        # MLP for classification
        self.refine_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128),  # Concatenate local+global+attended features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, batch):
        # Extract local features with EdgeConv
        local_features = self.edgeconv1(x, batch)  
        local_features = self.edgeconv2(local_features, batch)  

        # Compute global features
        global_features = torch.max(local_features, dim=1, keepdim=True)[0]  
        global_features = global_features.repeat(1, x.size(1), 1)  

        # Apply Self-Attention
        attended_features = self.attention(local_features)  

        # Concatenate features
        combined_features = torch.cat([local_features, global_features, attended_features], dim=-1)

        # MLP for classification
        output = self.refine_mlp(combined_features)

        return F.log_softmax(output, dim=-1)

