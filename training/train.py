from data.SemanticKittiDataset import SemanticKittiDataset, semantic_kitti_collate_fn

from args.args import Args
from transforms.random_downsample import RandomDownsample
from util.run_config import RunConfig
from util.checkpoint import save_checkpoint, save_best, load_checkpoint
from models.pn_linear import get_model
from training.validate import validate

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from torchmetrics import JaccardIndex

from tqdm import tqdm
import wandb


def get_optimizer_memory_size(optimizer):
    memory_size = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            # Add the size of the parameter tensor in bytes
            memory_size += param.numel() * param.element_size()
    return memory_size


def train_epoch(epoch, epochs, model, optimizer, loss_fn, train_dataloader):
    model.train()
    batch_iou = JaccardIndex(task="multiclass", num_classes=20).to(Args.args.device)

    epoch_loss = 0
    epoch_iou = 0
    num_batches = 0
    avg_mem_usage = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            # check batch size isnt 1 to avoid batch norm layers crashing
            if data.size(0) == 1:
                continue # so batch norm dosnt shit the bed

            tepoch.set_description(f"Epoch {epoch}/{epochs}")

            data, target = data.to(Args.args.device), target.to(Args.args.device)
            optimizer.zero_grad()

            y_pred = model(data)
            logits = y_pred.permute(0, 2, 1)
            loss = loss_fn(logits, target)

            preds = torch.argmax(logits, dim=1)

            with torch.no_grad():
                iou_score = batch_iou(preds.detach(), target.detach()).item()

            epoch_loss += loss.item()
            epoch_iou += iou_score
            num_batches += 1
            
            loss.backward()
            optimizer.step()

            # memory optimization
            del data, target, y_pred, logits, preds

            if Args.args.device == torch.device('mps'):
                mem_usage = torch.mps.driver_allocated_memory()/ 1e9
                op_mem_usage = get_optimizer_memory_size(optimizer)/1e6
                avg_mem_usage += mem_usage

            # tqdm progress bar
            if Args.args.device == torch.device('mps'):
                tepoch.set_postfix(epoch_loss=epoch_loss/(batch_idx+1),
                                   epoch_iou=100. * (epoch_iou / num_batches),
                                   mem_usage=f'{mem_usage:.2f}GB',
                                   avg_mem_usage=f'{avg_mem_usage / num_batches:.2f}GB',
                                   op_mem_usage=f'{op_mem_usage:.2f}MB')
            else:
                tepoch.set_postfix(epoch_loss=epoch_loss/(batch_idx+1),
                                   epoch_iou=100. * (epoch_iou / num_batches))


            if Args.args.wandb and batch_idx % Args.run_config.batch_log_rate == 0:
                if Args.args.device == torch.device('mps'):
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_iou': iou_score,
                        'mem_usage_gb': mem_usage,
                        'avg_mem_usage_gb': (avg_mem_usage / num_batches),
                        'optimizer_mem_usage_mb': op_mem_usage,
                        'batch_idx': batch_idx
                    })
                else:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_iou': iou_score,
                        'batch_idx': batch_idx
                    })

    epoch_loss /= num_batches
    epoch_iou /= num_batches

    return epoch_loss, epoch_iou

def train():
    run_config = Args.run_config
    if Args.args.verbose:
        print(f"Beginning Run {run_config.run_id}")

    # add RandomDownsample to the trainset for feasable computation
    train_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, downsample=True, split='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=run_config.train_batch_size,
                                  num_workers=run_config.num_workers,
                                  persistent_workers=True,
                                  pin_memory=False,
                                  shuffle=True)
    if Args.args.validate:
        valid_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.config, downsample=False, split='valid')
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=run_config.valid_batch_size,
                                      num_workers=run_config.num_workers,
                                      persistent_workers=True,
                                      pin_memory=False,
                                      shuffle=True,
                                      collate_fn=semantic_kitti_collate_fn)

    if Args.args.verbose:
        print("Loaded datasets")

    model = get_model(num_classes=train_dataset.num_classes).to(Args.args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr)
    loss = F.nll_loss

    epoch_offset = 1
    if Args.args.from_checkpoint:
        epoch_offset += load_checkpoint(model, optimizer, Args.args.from_checkpoint)

    if Args.args.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Loaded Model:")
        print(f"\tTrainable parameters: {trainable_params}")

    for epoch in range(epoch_offset, run_config.epochs+epoch_offset):
        epoch_loss, epoch_iou = train_epoch(epoch, run_config.epochs + epoch_offset, model, optimizer, loss, train_dataloader)

        if Args.args.validate:
            if Args.args.verbose:
                print("Epoch complete, validating...")
            val_loss, val_iou = validate(model, valid_dataloader, loss)

            if Args.args.verbose:
                print(f"Validation complete: val_loss: {val_loss} - val_iou: {val_iou}")

            if Args.args.wandb:
                wandb.log({
                    'epoch_loss': epoch_loss,
                    'epoch_iou': epoch_iou,
                    'val_loss': val_loss,
                    'val_iou': val_iou,
                    'epoch': epoch
                })
        else:
            val_loss, val_iou = 0, 0
            if Args.args.wandb:
                wandb.log({
                    'epoch_loss': epoch_loss,
                    'epoch_iou': epoch_iou,
                    'epoch': epoch
                })

        if Args.args.checkpoint != 0 and epoch % Args.args.checkpoint == 0:
            save_checkpoint(model, optimizer, epoch, val_iou)
