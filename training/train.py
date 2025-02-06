from data.SemanticKittiDataset import SemanticKittiDataset
from torch.utils.data import DataLoader
from args.args import Args

def load_trainset():
    train_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, transform=None, train=True)
    return DataLoader(train_dataset, batch_size=4, num_workers=4, pin_memory=False)

def load_testset():
    test_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, transform=None, train=False)
    return DataLoader(test_dataset, batch_size=4, num_workers=4, pin_memory=False)

def train():
    trainset = load_trainset()

    if Args.args.verbose:
        print("Loaded train dataset")
