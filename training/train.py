from data.SemanticKittiDataset import SemanticKittiDataset, semantic_kitti_collate_fn
from training.run_config import RunConfig
# from models.PointNetLoss import pointnet_segmentation_loss
# from models.PointNetSegmentation import PointNetSegmentation
import torch.nn.functional as F
from models.eg_pointnet import get_model, get_loss
from args.args import Args
from util.checkpoint import save_checkpoint, save_best

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import JaccardIndex



def train_epoch(epoch, epochs, model, optimizer, loss_fn, train_dataloader):
    model.train()

    epoch_loss = 0
    epoch_iou = JaccardIndex(task="multiclass", num_classes=20).to(Args.args.device)
    epoch_iou.reset()

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch_idx, (data, target, _) in enumerate(tepoch):
            # check batch size isnt 1 to avoid batch norm layers crashing
            if data.size(0) == 1:
                continue

            tepoch.set_description(f"Epoch {epoch}/{epochs}")

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

    epoch_offset = 1
    if Args.args.from_checkpoint:
        epoch_offset += load_checkpoint(model, optimizer, Args.args.from_checkpoint)

    if Args.args.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer_state_size = sum([param.numel() * param.element_size() for param in optimizer.state.values()])
        print("Loaded Model:")
        print(f"\tTrainable parameters: {trainable_params}")
        print(f"\tOptimizer memory usage: {optimizer_state_size / 1e6} MB")

    for epoch in range(epoch_offset, run_config.epochs+epoch_offset):
        train_epoch(epoch, run_config.epochs, model, optimizer, loss, train_dataloader)

        if Args.args.checkpoint != 0 and epoch % Args.args.checkpoint == 0:
            save_checkpoint(model, optimizer, epoch, 0, run_config, False)
