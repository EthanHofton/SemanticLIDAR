from data.SemanticKittiDataset import SemanticKittiDataset, semantic_kitti_collate_fn
from training.run_config import RunConfig
# from models.PointNetLoss import pointnet_segmentation_loss
# from models.PointNetSegmentation import PointNetSegmentation
import torch.nn.functional as F
from models.eg_pointnet import get_model, get_loss
from args.args import Args

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import JaccardIndex


def train_epoch(epoch, model, optimizer, loss_fn, train_dataloader):
    model.train()

    epoch_loss = 0
    epoch_iou = JaccardIndex(task="multiclass", num_classes=20).to(Args.args.device)
    epoch_iou.reset()

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch_idx, (data, target, _) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(Args.args.device), target.to(Args.args.device)
            optimizer.zero_grad()

            y_pred, _ = model(data)
            logits = y_pred.permute(0, 2, 1)
            loss = loss_fn(logits, target)

            preds = torch.argmax(logits, dim=1)

            epoch_iou.update(preds, target)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(epoch_loss=epoch_loss/(batch_idx+1), epoch_iou=100. * epoch_iou.compute().item())

    epoch_loss /= len(train_dataloader)
            

def train():
    run_config = RunConfig(epochs=5, lr=1e-3)

    if Args.args.verbose:
        print(f"Beginning Run {run_config.run_id}")

    train_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, transform=None, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=8, pin_memory=False, collate_fn=semantic_kitti_collate_fn)

    if Args.args.verbose:
        print("Loaded datasets")

    # model = PointNetSegmentation(num_classes=train_dataset.num_classes).to(Args.args.device)
    model = get_model(num_class=train_dataset.num_classes).to(Args.args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr)
    loss = F.nll_loss

    if Args.args.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Loaded Model. Trainable parameters: {trainable_params}")

    for epoch in range(1, run_config.epochs+1):
        train_epoch(epoch, model, optimizer, loss, train_dataloader)
