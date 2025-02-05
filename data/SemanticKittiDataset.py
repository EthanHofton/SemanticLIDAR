from torch.utils.data import Dataset

class SemanticKittiDataset(Dataset):

    def __init__(self, ds_path, config_path):
        self.ds_path = ds_path


    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
