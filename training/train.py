from models.PointNetSegmentation import PointNetSegmentation
from data.SemanticKittiDataset import SemanticKittiDataset
from training.run_config import RunConfig
from models.PointNetLoss import pointnet_segmentation_loss
from args.args import Args

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(epoch, model, optimizer, train_dataloader):
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0
    counter = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            batch_loss = 0
            batch_acc = 0

            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(Args.args.device), target.to(Args.args.device)
            optimizer.zero_grad()

            y_pred = model(data)
            loss = pointnet_segmentation_loss(y_pred, target)

            predictions = torch.argmax(y_pred, dim=1)
            correct = (predictions == target).sum().item()

            # batch loss and acc
            batch_loss = loss.item()
            batch_acc = correct / target.size(0)

            epoch_loss += batch_loss
            epoch_acc += correct
            total_samples += target.size(0)
            
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(epoch_loss=epoch_loss/(batch_idx+1), epoch_acc=100. * (epoch_acc/total_samples))

    epoch_loss /= len(train_dataloader)
    epoch_acc /= total_samples
            

def train():
    run_config = RunConfig(epochs=5, lr=1e-3)

    if Args.args.verbose:
        print(f"Beginning Run {run_config.run_id}")

    train_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, transform=None, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, pin_memory=False)

    if Args.args.verbose:
        print("Loaded datasets")

    model = PointNetSegmentation(num_classes=train_dataset.num_classes).to(Args.args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr)

    if Args.args.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Loaded Model. Trainable parameters: {trainable_params}")

    for epoch in range(1, run_config.epochs+1):
        train_epoch(epoch, model, optimizer, train_dataloader)
