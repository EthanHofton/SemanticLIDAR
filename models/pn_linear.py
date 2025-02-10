import torch
import torch.nn as nn
import torch.nn.functional as F

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        
        # Shared MLP for each point
        self.shared_mlp = nn.Sequential(
            nn.Linear(3, 64),  # Input dimension is 3 (x, y, z)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Max pooling layer (to aggregate global features)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # MLP for the point features after global aggregation
        self.refine_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128),  # Concatenate the global and point features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output for each point: num_classes
        )

    def forward(self, x):
        """
            get_model forward pass

            input:
                - x: the data in shape (batch_size, N, 3)
            output:
                - output: the semantic segmentation of the input of size (batch_size, N, num_classes)
        """
        # Input x is of shape (batch_size, N, 3)
        batch_size, N, _ = x.size()
        
        # Shared MLP for each point (point-wise features)
        point_features = self.shared_mlp(x)  # Shape: (batch_size, N, 128)
        
        # Global feature aggregation using max pooling
        global_features = torch.max(point_features, dim=1, keepdim=True)[0]  # Shape: (batch_size, 1, 128)
        
        # Repeat global features for each point in the batch
        global_features = global_features.repeat(1, N, 1)  # Shape: (batch_size, N, 128)
        
        # Concatenate point-wise and global features
        point_features = torch.cat([point_features, global_features], dim=-1)  # Shape: (batch_size, N, 256)
        
        # Refine the point features
        output = self.refine_mlp(point_features)  # Shape: (batch_size, N, num_classes)
        
        return output
