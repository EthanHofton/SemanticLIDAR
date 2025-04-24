from args.parse_args import parse_args
from args.args import Args

from util.visualize import visualize
from util.make_dataset import make_dataset

from training.train import train
from training.validate import validate

def run_commands(help):
    if Args.args.command == 'visualize':
        visualize()
    elif Args.args.command == 'train':
        train()
    elif Args.args.command == 'make_dataset':
        make_dataset()
    elif Args.args.command == 'validate':
        validate()
    else:
        help()

def get_transforms(train):
    transforms = []

    transforms.append(T.BatchedDownsample())
    transforms.append(T.NpToTensor())

    return T.Compose(transforms)


if __name__ == "__main__":
    args, print_help = parse_args()
    Args.init(args)
    run_commands(print_help)

    # from util.checkpoint import load_model
    # from models.pn_linear import get_model
    # from data.SemanticKittiDataset import SemanticKittiDataset, semantic_kitti_collate_fn
    # import transforms.transforms as T
    # from torch.utils.data import DataLoader
    # import torch
    # 
    # run_config = Args.run_config
    # model = get_model(20)
    # train_dataset = SemanticKittiDataset(ds_path=Args.args.dataset, ds_config=Args.args.ds_config, transform=get_transforms(True), split='train')
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=run_config.train_batch_size,
    #                               num_workers=run_config.num_workers,
    #                               persistent_workers=True,
    #                               pin_memory=True,
    #                               shuffle=True,
    #                               collate_fn=T.bds_collate_fn)
    # train_iter = iter(train_dataloader)
    # batch = next(train_iter)
    # dummy_input = batch[0]
    #
    # onnx_file = "model.onnx"
    # torch.onnx.export(model, dummy_input, onnx_file, input_names=["LiDAR Scan"], output_names=["Semantic Segmentation"])
