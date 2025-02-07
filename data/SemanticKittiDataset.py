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

        scan = SemLaserScan(self.config['color_map'], project=False)
        scan.open_scan(scan_file)
        scan.open_label(label_file)

        if self.transform:
            scan = self.transform(scan)

        mapped_labels = torch.tensor([self.config["learning_map"][label] for label in scan.sem_label], dtype=torch.long)
        points = torch.tensor(scan.points, dtype=torch.float32).transpose(0, 1)

        return points, mapped_labels

    def __len__(self):
        return self.sample_size


# TODO: allow batching via masking
def semantic_kitti_collate_fn(batch):
    point_clouds, labels = zip(*batch)
    
    # Find the maximum number of points in the batch
    max_len = max([pc.size(0) for pc in point_clouds])
    
    padded_point_clouds = []
    point_cloud_masks = []  # To keep track of the valid points (non-padded)
    
    for pc in point_clouds:
        padding = torch.zeros((max_len - pc.size(0), pc.size(1)))  # Assuming (N, 3) shape for point clouds (x, y, z)
        padded_pc = torch.cat([pc, padding], dim=0)  # Pad the point cloud
        
        # Create a mask where 1 indicates a valid point, and 0 indicates padding
        mask = torch.ones(max_len)
        mask[pc.size(0):] = 0  # Set the mask values to 0 for padded points
        
        padded_point_clouds.append(padded_pc)
        point_cloud_masks.append(mask)
    
    # Stack the point clouds and masks
    point_clouds_batch = torch.stack(padded_point_clouds)
    masks_batch = torch.stack(point_cloud_masks)
    labels_batch = torch.tensor(labels)

    return point_clouds_batch, masks_batch, labels_batch
