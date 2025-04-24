from args.args import Args
import os
import torch
import wandb

def save_checkpoint(model, optimizer, epoch, val_iou):
    run_config = Args.run_config
    checkpoint_path = f'run-{run_config.run_id}-checkpoints'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint={
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou
    }

    filename = f"{run_config.run_id}-epoch-{epoch}.pth"
    checkpoint_filename = os.path.join(checkpoint_path, filename)
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved at epoch {epoch}, with validation mIoU: {val_iou} ({checkpoint_filename})")

    if Args.args.wandb:
        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(checkpoint_filename)
        wandb.log_artifact(artifact)


def save_best(model, optimizer, epoch, metric):
    run_config = Args.run_config

    checkpoint_path = f'run-{run_config.run_id}-checkpoints'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint={
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
        'epoch': epoch
    }

    filename = f"{run_config.run_id}-best-{metric}-epoch-{epoch}.pth"
    checkpoint_filename = os.path.join(checkpoint_path, filename)
    torch.save(checkpoint, checkpoint_filename)
    print(f"Best model for {metric} saved ({checkpoint_filename})")

    if Args.args.wandb:
        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(checkpoint_filename)
        wandb.log_artifact(artifact)


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded ({checkpoint_path}): Resuming from epoch {epoch}")
        return epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0

def load_model(model, checkpoint_path, device='cpu'):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded: {checkpoint_path}")
    else:
        raise Exception(f"Model ({checkpoint_path}) not found!")

