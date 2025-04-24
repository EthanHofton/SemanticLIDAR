import torch
import torch.nn as nn
import torch.nn.functional as F

class get_model_conv(nn.Module):
    def __init__(self, num_classes):
        super(get_model_conv, self).__init__()
        
        # Shared MLP for each point
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),  # Input dimension is 3 (x, y, z)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # MLP for the point features after global aggregation
        self.refine_mlp = nn.Sequential(
            nn.Conv1d(2048, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, kernel_size=1)
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
        x = x.permute(0, 2, 1) # Match Conv1D expected format (batch_size, 3, N)
        point_features = self.shared_mlp(x)  # Shape: (batch_size, 1024, N)
        
        # Global feature aggregation using max pooling
        global_features = torch.max(point_features, dim=2, keepdim=True)[0]  # Shape: (batch_size, 1024, 1)
        
        # Repeat global features for each point in the batch
        global_features = global_features.repeat(1, 1, N)  # Shape: (batch_size, 1024, N)
        
        # Concatenate point-wise and global features
        point_features = torch.cat([point_features, global_features], dim=1)  # Shape: (batch_size, 2048, N)
        
        # Refine the point features
        output = self.refine_mlp(point_features)  # Shape: (batch_size, num_classes, N)
        output = output.permute(0, 2, 1) # Shape: (batch_size, N, num_classes)

        log_probs = F.log_softmax(output, dim=-1) # return softmax for nll
        
        return log_probs