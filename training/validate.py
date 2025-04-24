import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import wandb
import numpy as np
from tqdm import tqdm
from torchmetrics import JaccardIndex, ConfusionMatrix
from args.args import Args
import torch
from models.pn_conv import get_model_conv
import torch.nn.functional as F
from data.SemanticKittiDataset import SemanticKittiDataset
from transforms.transforms import *
from torch.utils.data import DataLoader
from util.checkpoint import load_model

def display(confusion_matrix, IoU):
    labels = Args.args.ds_config['labels']
    labels_map = Args.args.ds_config['learning_map_inv']
    ignored_classes = Args.args.ds_config['learning_ignore']
    num_classes = len(labels_map)
    
    valid_class_indices = [i for i in range(num_classes) if not ignored_classes[i]]
    class_labels = [labels[labels_map[i]] for i in valid_class_indices]

    iou_values = IoU.compute().cpu()
    filtered_iou = [iou_values[i] for i in valid_class_indices]

    # --- Per-class IoU Plot ---
    fig_iou, ax = plt.subplots(figsize=(10, 6))
    ax.bar(class_labels, filtered_iou)
    ax.set_title("Per-class IoU")
    ax.set_xlabel("Class")
    ax.set_ylabel("IoU")
    fig_iou.tight_layout()
    plt.show()

    # --- Confusion Matrix Plot ---
    conf_matrix = confusion_matrix.compute()
    conf_matrix = conf_matrix.float() / (conf_matrix.sum(dim=1, keepdim=True) + 1e-5) * 100
    conf_matrix = conf_matrix.cpu().numpy()
    filtered_conf_matrix = conf_matrix[np.ix_(valid_class_indices, valid_class_indices)]

    fig_conf, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(filtered_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig_conf.tight_layout()
    plt.show()

    return fig_iou, fig_conf
    

def log(val_mIoU, val_loss, epoch_iou, epoch_loss, epoch, confusion_matrix, IoU, disp=True):
    if disp:
        fig_iou, fig_conf = display(confusion_matrix, IoU)

    # --- Log to wandb ---
    if Args.args.wandb:
        wandb.log({
            "epoch": epoch,
            "epoch_iou": epoch_iou,
            "epoch_loss": epoch_loss,
            "val_mIoU": val_mIoU,
            "val_loss": val_loss,
            "Per-class IoU": wandb.Image(fig_iou),
            "Confusion Matrix": wandb.Image(fig_conf)
        })
    else:
        print(f"Epoch {epoch}:")
        print(f"  - Per-class IoU: {val_mIoU * 100:.2f}%")
        print(f"  - Validation loss: {val_loss:.4f}")

    if disp:
        plt.close(fig_iou)
        plt.close(fig_conf)


def val(model, valid_dataloader, loss_fn):
    model.eval()
    num_classes = len(Args.args.ds_config['learning_map_inv'])
    device = Args.args.device
    
    mIoU = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)
    IoU = JaccardIndex(task="multiclass", num_classes=num_classes, average='none', ignore_index=0).to(device)
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)
    
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(valid_dataloader, unit='batch') as tbatch:
            for batch_idx, (data, target) in enumerate(tbatch):
                tbatch.set_description(f'Batch {batch_idx}/{len(valid_dataloader)}')
                data, target = data.to(device), target.to(device)
                y_pred = model(data) # shape (batch_size, N, num_classes)
                logits = y_pred.permute(0, 2, 1) # shape (batch_size, num_classes, N)
    
                preds = torch.argmax(logits, dim=1) # shape (batch_size, N)
                val_loss += loss_fn(logits, target, ignore_index=0).item()
                IoU.update(preds, target)
                mIoU.update(preds, target)
                num_batches += 1
    
                confusion_matrix.update(preds.flatten(), target.flatten())
    
                tbatch.set_postfix(val_loss=val_loss/num_batches,
                                   val_mIoU=mIoU.compute().item() * 100)
    
    val_loss /= num_batches
    val_mIoU = mIoU.compute().item()

    return val_mIoU, val_loss, confusion_matrix, IoU

def cli_validate():
    """
        Validate the model on the validation set.
    """
    if Args.args.verbose:
        print("Creating model")
    model = get_model_conv(len(Args.args.ds_config['learning_map_inv']))
    if Args.args.verbose:
        print("Opening model")
    load_model(model, Args.args.model, device=Args.args.device)
    if Args.args.verbose:
        print("Convering to device")
    model.to(Args.args.device)
    if Args.args.verbose:
        print("Model loaded")

    # Load the loss function
    loss_fn = F.nll_loss

    transforms = []
    transforms.append(BatchedDownsample(Args.run_config.downsample, Args.run_config.mini_batch_size))
    transforms.append(NpToTensor())

    t = Compose(transforms)

    # Load the validation dataloader
    if Args.args.verbose:
        print("Loading validation dataset")
    valid_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.ds_config, transforms=t, split='valid')
    valid_dataloader = DataLoader(valid_dataset,
                                batch_size=Args.run_config.batch_size,
                                num_workers=Args.run_config.num_workers,
                                persistent_workers=False,
                                pin_memory=True,
                                shuffle=True,
                                collate_fn=bds_collate_fn)

    if Args.args.verbose:
        print("Validating...")
    # Validate the model
    val_mIoU, val_loss, confusion_matrix, IoU = val(model, valid_dataloader, loss_fn)

    if Args.args.verbose:
        print("Logging/Displaying results...")
    # Log the results
    fig_iou, fig_conf = display(confusion_matrix, IoU)
    print(f"Validation mIoU: {val_mIoU * 100:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")

    plt.close(fig_iou)
    plt.close(fig_conf)
