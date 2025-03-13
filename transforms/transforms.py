import numpy as np
import torch
import open3d as o3d
import time


class RandomDownsample:
    def __init__(self, max_points=1024):
        self.max_points = max_points

    def __call__(self, points, labels):
        if points.shape[0] > self.max_points:
            idxs = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[idxs]
            labels = labels[idxs]
        elif points.shape[0] < self.max_points:
            padding = np.zeros((self.max_points - points.shape[0], 3))  # Padding for 3D points
            points = np.vstack([points, padding])
            labels = np.hstack([labels, np.zeros(self.max_points - labels.shape[0])])  # Pad labels accordingly

class BatchedDownsample:
    def __init__(self, max_points=1024, batches=32):
        self.max_points = max_points
        self.batches = batches

    def __call__(self, points, labels):
        point_batches = np.zeros((self.batches, self.max_points, 3))
        label_batches = np.zeros((self.batches, self.max_points,))
        for i in range(self.batches):
            if points.shape[0] > self.max_points:
                idxs = np.random.choice(points.shape[0], self.max_points, replace=False)
                point_batches[i] = points[idxs]
                label_batches[i] = labels[idxs]
            elif points.shape[0] < self.max_points:
                padding = np.zeros((self.max_points - points.shape[0], 3))  # Padding for 3D points
                point_batches[i] = np.vstack([points, padding])
                label_batches[i] = np.hstack([labels, np.zeros(self.max_points - labels.shape[0])])

        return point_batches, label_batches

def bds_collate_fn(batch):
    """
    Stack mini-batches

    Converts batch: (batch_size, mini_batch_size, max_points, 3)
    to point_batches, label_batches:
    (batch_size * mini_batch_size, max_points, 3) points
    (batch_size * mini_batch_size, max_points,) lables
    -> stacks mini_batches across batches
    
    """
    points, labels = zip(*batch)
    points = torch.stack(points)
    labels = torch.stack(labels)
    point_batches = torch.zeros((points.shape[0] * points.shape[1], points.shape[2], points.shape[3]), dtype=torch.float32)
    label_batches = torch.zeros((labels.shape[0] * labels.shape[1], labels.shape[2],), dtype=torch.long)
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            point_batches[i * j] = points[i][j]

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label_batches[i * j] = labels[i][j]

    return point_batches, label_batches


class StratifiedDownsample:
    def __init__(self, max_points=1024):
        self.max_points = max_points

    def __call__(self, points, labels):
        unique_labels = np.unique(labels)
        total_points = points.shape[0]

        # Calculate the number of points per label class
        class_counts = {label: np.sum(labels == label) for label in unique_labels}
        class_proportions = {label: count / total_points for label, count in class_counts.items()}

        # List to store downsampled points and labels
        downsampled_points = []
        downsampled_labels = []

        # Iterate over each label class and sample proportionally
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_count = class_counts[label]
            sampled_count = int(np.round(class_proportions[label] * self.max_points))

            # If the label class has more points than required, randomly downsample
            if label_count > sampled_count:
                sampled_indices = np.random.choice(label_indices, sampled_count, replace=False)
            else:
                sampled_indices = label_indices  # No downsampling needed, just use all

            downsampled_points.append(points[sampled_indices])
            downsampled_labels.append(labels[sampled_indices])

        # Concatenate the downsampled points and labels
        downsampled_points = np.vstack(downsampled_points)
        downsampled_labels = np.hstack(downsampled_labels)

        # If the total number of points is less than max_points, pad with zeros
        if downsampled_points.shape[0] < self.max_points:
            padding = np.zeros((self.max_points - downsampled_points.shape[0], 3))  # Padding for 3D points
            downsampled_points = np.vstack([downsampled_points, padding])
            downsampled_labels = np.hstack([downsampled_labels, np.zeros(self.max_points - downsampled_labels.shape[0])])

        return downsampled_points, downsampled_labels


class FurthestPointDownsample:
    def __init__(self, num_points=1024):
        """
        Args:
            voxel_size (float): The size of the voxel grid for downsampling.
        """
        self.num_points = num_points

    def _farthest_point_sampling(self, points, num_points):
        n_points = len(points)
        
        # Initialize distances to a large value (inf)
        distances = np.full(n_points, np.inf)
        
        # Randomly select the first point
        farthest_indices = [np.random.choice(n_points)]
        
        # Calculate the distance from all points to the first point
        distances[:] = np.linalg.norm(points - points[farthest_indices[0]], axis=1)
        
        # Vectorized farthest point sampling
        for _ in range(1, num_points):
            # Find the farthest point based on the maximum distance
            farthest_index = np.argmax(distances)
            farthest_indices.append(farthest_index)
            
            # Update the distance to the nearest farthest point
            new_distances = np.linalg.norm(points - points[farthest_index], axis=1)
            distances = np.minimum(distances, new_distances)
        
        return farthest_indices

    def __call__(self, points, labels):
        start = time.time()
        indicies = self._farthest_point_sampling(points, self.num_points)
        end = time.time()

        print(f"FPS sample took {end - start}")

        points = points[indicies]
        labels = labels[indicies]

        return points, labels

class NpToTensor:
    def __init__(self):
        pass

    def __call__(self, points, labels):
        """
        Convert o3d Point Cloud, numpy array to Tensors
        """
        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class Open3dPointCloudToTensor:
    def __init__(self):
        pass

    def __call__(self, pcd, labels):
        """
        Convert o3d Point Cloud, numpy array to Tensors
        """
        return torch.tensor(np.asarray(pcd.points), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class ToOpen3dPointCloud():
    def __init__(self):
        pass

    def __call__(self, points, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return points, labels


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, labels):
        for t in self.transforms:
            points, labels = t(points, labels)

        return points, labels
