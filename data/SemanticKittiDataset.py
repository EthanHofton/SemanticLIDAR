from util.auxiliary.laserscan import SemLaserScan
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from args.args import Args
from glob import glob
import os


class SemanticKittiDataset(Dataset):

    def __init__(self, ds_path, ds_config, transform=None, train=True, split_ratio=0.8, seed=42):
        """
            Semantic KITTI dataset
            - use train=True for training
            ** Ensure that split_ratio and seed remain the same for train=True and train=False**
            - ds_path is the path to the semantic kitti dataset in the layout required from the website.
            - cs_config is the config file from the SemantiKITTIApi
        """
        self.ds_path = ds_path
        self.transform = transform
        self.config = ds_config
        self.color_dict = self.config['color_map']

        # self.scan_files = sorted([file for file in os.walk(ds_path) if file.endswith('.bin')])
        # self.label_files = sorted([file for file in os.walk(ds_path) if file.endswith('.label')])
        self.scan_files = sorted(glob(os.path.join(ds_path, "**/*.bin"), recursive=True))
        self.label_files = sorted(glob(os.path.join(ds_path, "**/*.label"), recursive=True))

        if Args.args.verbose:
            print(f"Found {len(self.scan_files)} scan files")
            print(f"Found {len(self.label_files)} label files")

        if len(self.scan_files) != len(self.label_files):
            raise Exception(f'Scan file ({len(self.scan_files)}) and Label file ({len(self.label_files)}) size missmatch')

        full_sample_size = len(self.scan_files)

        # Split into train/test sets
        train_scans, test_scans, train_labels, test_labels = train_test_split(
            self.scan_files, self.label_files, train_size=split_ratio, random_state=seed
        )

        # Select the appropriate subset
        self.scan_files = train_scans if train else test_scans
        self.label_files = train_labels if train else test_labels

        self.sample_size = len(self.scan_files)

        if Args.args.verbose:
            dtype = "training" if train else "testing"
            print(f"Loaded {dtype} set of sample size {self.sample_size}/{full_sample_size} ({split_ratio} split) from {self.ds_path}")


    def __getitem__(self, idx):
        scan_file = self.scan_files[idx]
        label_file = self.label_files[idx]

        seq_labels = os.path.basename(os.path.dirname(os.path.dirname(label_file.path)))  # extract the sequence folder
        scan_labels = os.path.basename(os.path.splitext(label_file.name)[0])  # extract the label name

        seq_scan = os.path.basename(os.path.dirname(os.path.dirname(scan_file.path)))  # extract the sequence folder
        scan_scan = os.path.basename(os.path.splitext(scan_file.name)[0])  # extract the scan name

        if seq_labels == seq_scan and scan_labels == scan_scan:
            raise Exception('Error loading labels for scan {scan_file}. Scan label ({scan_scan}) must match label label ({scan_labels}) and must be of same sequence ({seq_scan} != {seq_labels})')

        scan = SemLaserScan(self.color_dict, project=False)
        scan.open_scan(scan_file)
        scan.open_label(label_file)

        if self.transform:
            scan = self.transform(scan)

        return torch.tensor(scan.points, dtype=torch.float32), torch.tensor(scan.sem_label, dtype=torch.long)

    def __len__(self):
        return self.sample_size
