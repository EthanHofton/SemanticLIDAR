from util.auxiliary.laserscan import SemLaserScan
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from args.args import Args
from glob import glob
import torch
import os
import numpy as np


class SemanticKittiDataset(Dataset):

    def __init__(self, ds_path, ds_config, transform=None, split='train'):
        """
            Semantic KITTI dataset
            - use train=True for training
            - ds_path is the path to the semantic kitti dataset in the layout required from the website.
            - cs_config is the config file from the SemantiKITTIApi
        """
        if split not in ['test', 'train', 'valid']:
            raise Exception(f'split must be either test, train or valid. Not {split}')

        self.ds_path = ds_path
        self.transform = transform
        self.config = ds_config
        self.split = split
        self.has_labels = (self.split == 'train' or self.split == 'valid')
        # self.scan = SemLaserScan(self.config['color_map'], project=False)

        sequences = [os.path.join(ds_path, 'sequences', f'{int(sequence):02}')
                     for sequence in self.config['split'][split]
                    ]
        self.scan_files = []
        for path in sequences:
            self.scan_files.extend(glob(os.path.join(path, "**/*.bin"), recursive=True))
        self.scan_files = sorted(self.scan_files)

        if self.has_labels:
            self.label_files = []
            for path in sequences:
                self.label_files.extend(glob(os.path.join(path, "**/*.label"), recursive=True))
            self.label_files = sorted(self.label_files)

        self.num_classes = len(self.config['learning_map_inv'])

        # temporarly take a portion of the dataset for development. Un comment to take an 8th of the dataset
        # self.scan_files = self.scan_files[:len(self.scan_files) // 8]
        # self.label_files = self.label_files[:len(self.label_files) // 8]

        if Args.args.verbose:
            print(f"Found {len(self.scan_files)} scan files")
            print(f"Found {len(self.label_files)} label files")

        if len(self.scan_files) != len(self.label_files):
            raise Exception(f'Scan file ({len(self.scan_files)}) and Label file ({len(self.label_files)}) size missmatch')

        self.sample_size = len(self.scan_files)

        if Args.args.verbose:
            print(f"Loaded {self.split} set of sample size {self.sample_size} from {self.ds_path}")


    def __getitem__(self, idx):
        scan_file = self.scan_files[idx]
        label_file = self.label_files[idx]

        # get points
        points_scan = np.fromfile(scan_file, dtype=np.float32)
        points_scan = points_scan.reshape((-1, 4))
        points = points_scan[:, 0:3] # get xyz
        points = torch.tensor(points, dtype=torch.float32).transpose(0, 1)

        # get labels
        labels = np.fromfile(label_file, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF  # semantic label in lower half
        labels = torch.tensor([self.config["learning_map"][label] for label in labels], dtype=torch.long)

        return points, labels


    def __len__(self):
        return self.sample_size


def semantic_kitti_collate_fn(batch):
    """
    Custom collate function to pad point clouds and labels, and create masks.
    """
    points, labels = zip(*batch)
    
    # Determine max number of points in the batch
    max_points = max([pc.size(1) for pc in points])
    
    padded_points = torch.zeros(len(batch), 3, max_points)  # Shape: (batch_size, 3, max_points)
    padded_labels = torch.zeros(len(batch), max_points, dtype=torch.long)  # Shape: (batch_size, max_points)
    
    for i, (point_cloud, label) in enumerate(zip(points, labels)):
        num_points = point_cloud.size(1)
        
        # Pad the points
        padded_points[i, :, :num_points] = point_cloud
        
        # Pad the labels
        padded_labels[i, :num_points] = label
    
    return padded_points, padded_labels

