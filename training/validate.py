from data.SemanticKittiDataset import SemanticKittiDataset, semantic_kitti_collate_fn
from util.checkpoint import load_model
from models.pn_linear import get_model

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import JaccardIndex, ConfusionMatrix
from args.args import Args
from matplotlib import pyplot as plt
import seaborn as sns

def train_validate(model, valid_dataloader, loss_fn):
    model.eval()
    batch_iou = JaccardIndex(task="multiclass", num_classes=20).to(Args.args.device)

    val_loss = 0
    val_iou = 0
    num_batches = 0

    with torch.no_grad():
        with tqdm(valid_dataloader, unit='batch') as tbatch:
            for batch_idx, (data, target) in enumerate(tbatch):
                if data.size(0) == 1:
                    continue # so batch norm dosnt shit the bed

                tbatch.set_description(f'Batch {batch_idx}/{len(valid_dataloader)}')

                data, target = data.to(Args.args.device), target.to(Args.args.device)
                y_pred = model(data) # shape (batch_size, N, num_classes)
                # cross entroy loss needs shape: (batch_size, num_classes, N)
                logits = y_pred.permute(0, 2, 1)

                preds = torch.argmax(logits, dim=1)
                val_loss += loss_fn(logits, target).item()
                val_iou += batch_iou(preds, target).item()
                num_batches += 1

                if Args.args.device == torch.device('mps'):
                    tbatch.set_postfix(val_loss=val_loss/num_batches,
                       val_iou=100. * (val_iou / num_batches),
                       mem_usage=f'{(torch.mps.driver_allocated_memory()/ 1e9):.2f}GB')
                else:
                    tbatch.set_postfix(val_loss=val_loss/num_batches,
                       val_iou=100. * (val_iou / num_batches))


    val_loss /= num_batches
    val_iou /= num_batches

    return val_loss, val_iou


def validate():
    run_config = Args.run_config

    collate_fn = None
    if not (Args.args.downsample):
        collate_fn = semantic_kitti_collate_fn

    valid_dataset = SemanticKittiDataset(ds_path=Args.args.dataset,
                                         ds_config=Args.args.ds_config,
                                         downsample=Args.args.downsample,
                                         split='valid')
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=run_config.batch_size,
                                  num_workers=run_config.num_workers,
                                  persistent_workers=True,
                                  pin_memory=False,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    
    if Args.args.verbose:
        print('Loaded validate dataset')

    model = get_model(num_classes=valid_dataset.num_classes).to(Args.args.device)
    loss_fn = F.nll_loss

    load_model(model, Args.args.model)

    ## -- Model validation begins here --
    model.eval()
    num_classes = 20

    mIoU = JaccardIndex(task="multiclass", num_classes=num_classes).to(Args.args.device)
    if Args.args.per_class:
        IoU = JaccardIndex(task="multiclass", num_classes=num_classes, average='none').to(Args.args.device)
    if Args.args.confusion_matrix:
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(Args.args.device)

    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(valid_dataloader, unit='batch') as tbatch:
            for batch_idx, (data, target) in enumerate(tbatch):
                if data.size(0) == 1:
                    continue # so batch norm doesn't break

                tbatch.set_description(f'Batch {batch_idx}/{len(valid_dataloader)}')
                data, target = data.to(Args.args.device), target.to(Args.args.device)
                y_pred = model(data) # shape (batch_size, N, num_classes)
                logits = y_pred.permute(0, 2, 1) # shape (batch_size, num_classes, N)

                preds = torch.argmax(logits, dim=1) # shape (batch_size, N)
                val_loss += loss_fn(logits, target).item()
                if Args.args.per_class:
                    IoU.update(preds, target)
                mIoU.update(preds, target)
                num_batches += 1

                if Args.args.confusion_matrix:
                    confusion_matrix.update(preds.flatten(), target.flatten())

                if Args.args.device == torch.device('mps'):
                    tbatch.set_postfix(val_loss=val_loss/num_batches,
                                       val_mIoU=mIoU.compute().item(),
                                       mem_usage=f'{(torch.mps.driver_allocated_memory()/ 1e9):.2f}GB')
                else:
                    tbatch.set_postfix(val_loss=val_loss/num_batches,
                                       val_mIoU=mIoU.compute().item())

    val_loss /= num_batches
    val_mIoU = mIoU.compute().item()

    if Args.args.confusion_matrix:
        conf_matrix = confusion_matrix.compute()
        conf_matrix = conf_matrix.float() / (conf_matrix.sum(dim=1, keepdim=True) + 1e-5) * 100
        conf_matrix = conf_matrix.cpu().numpy()

    if Args.args.save:
        # save_results(conf_matrix, per_class_miou, mean_iou, output_dir)
        pass

    if Args.args.view:
        labels = Args.args.ds_config['labels']
        labels_map = Args.args.ds_config['learning_map_inv']

        class_labels = [labels[labels_map[index]] for index in range(num_classes)]
        # View the per-class IoU as a bar chart
        if Args.args.per_class:
            plt.figure(figsize=(10, 6))
            plt.bar(class_labels, IoU.compute().cpu().numpy())
            plt.title("Per-class IoU")
            plt.xlabel("Class Index")
            plt.ylabel("IoU")
            plt.show()

        # View the confusion matrix as a heatmap
        if Args.args.confusion_matrix:
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()

    # -- Display Results --
    print('Validation Results:')
    print(f'\tValidation Loss: {val_loss}')
    print(f'\tValidation mIoU: {val_mIoU}')
    if Args.args.per_class:
        print(f'\tPer-class IoU: {IoU.compute().cpu().numpy()}')
    if Args.args.confusion_matrix:
        print(f'\tConfusion Matrix: {conf_matrix}')

