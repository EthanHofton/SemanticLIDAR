import torch
from tqdm import tqdm
from torchmetrics import JaccardIndex
from args.args import Args

def validate(model, valid_dataloader, loss_fn):
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
                y_pred, _ = model(data)
                logits = y_pred.permute(0, 2, 1)

                preds = torch.argmax(logits, dim=1)
                val_loss += loss_fn(logits, target).item()
                val_iou += batch_iou(preds, target).item()
                num_batches += 1

                tbatch.set_postfix(val_loss=val_loss/num_batches,
                   val_iou=100. * (val_iou / num_batches),
                   mem_usage=f'{(torch.mps.driver_allocated_memory()/ 1e9):.2f}GB')

    val_loss /= num_batches
    val_iou /= num_batches

    return val_loss, val_iou

