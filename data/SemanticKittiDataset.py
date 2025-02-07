from util.auxiliary.laserscan import SemLaserScan
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
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

        mapped_labels = torch.tensor([self.config["learning_map"][label] for label in scan.sem_label], dtype=torch.long)

        if self.transform:
            scan = self.transform(scan)

        print(mapped_labels)
        exit()

        return torch.tensor(scan.points, dtype=torch.float32), mapped_labels

    def __len__(self):
        return self.sample_size
