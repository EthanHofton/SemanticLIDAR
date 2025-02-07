from data.SemanticKittiDataset import SemanticKittiDataset
from torch.utils.data import DataLoader
from args.args import Args
import torch

from models.PointNetSegmentation import PointNetSegmentation

def get_val_dataloader():
    val_dataloder = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, transform=None, split='valid')
    return DataLoader(test_dataset, batch_size=4, num_workers=4, pin_memory=False)

def train():
    train_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, transform=None, split='train')
    # train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=4, pin_memory=False)

    print(train_dataset.num_classes)
    exit()

    # optimizer = torch.optim.Adam()

    if Args.args.verbose:
        print("Loaded datasets")


