from util.auxiliary.laserscan import SemLaserScan
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from args.args import Args
from glob import glob
import torch
import os


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
        self.scan = SemLaserScan(self.config['color_map'], project=False)

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

        self.scan.open_scan(scan_file)
        self.scan.open_label(label_file)

        if self.transform:
            self.scan = self.transform(self.scan)

        mapped_labels = torch.tensor([self.config["learning_map"][label] for label in self.scan.sem_label], dtype=torch.long)
        points = torch.tensor(self.scan.points, dtype=torch.float32).transpose(0, 1)

        return points, mapped_labels

    def __len__(self):
        return self.sample_size


# def semantic_kitti_collate_fn(batch):
#     """
#     Custom collate function to pad point clouds and labels, and create masks.
#     """
#     points, labels = zip(*batch)
#     
#     # Determine max number of points in the batch
#     max_points = max([pc.size(1) for pc in points])
#     
#     padded_points = torch.zeros(len(batch), 3, max_points)  # Shape: (batch_size, 3, max_points)
#     padded_labels = torch.zeros(len(batch), max_points, dtype=torch.long)  # Shape: (batch_size, max_points)
#     mask = torch.zeros(len(batch), max_points)  # Shape: (batch_size, max_points)
#     
#     for i, (point_cloud, label) in enumerate(zip(points, labels)):
#         num_points = point_cloud.size(1)
#         
#         # Pad the points
#         padded_points[i, :, :num_points] = point_cloud
#         
#         # Pad the labels
#         padded_labels[i, :num_points] = label
#         
#         # Create a mask: 1 for real points, 0 for padding
#         mask[i, :num_points] = 1
#     
#     return padded_points, padded_labels, mask

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

