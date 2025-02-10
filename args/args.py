import yaml
import wandb
from util.get_device import get_device

class Args():
    args = {}

    @classmethod
    def init(cls, args):
        cls.args = args

        if getattr(cls.args, "device", None):
            # check device is avaliable on OS
            cls.args.device = get_device(cls.args.device)

        if getattr(cls.args, "wandb", None) != None:
            try:
                with open(cls.args.wandb, 'r') as file:
                    wandb_config = yaml.safe_load(file)
                if getattr(cls.args, "verbose", False):
                    print(f"Initalizing WandB with config file {cls.args.wandb}")
                wandb.init(config=wandb_config, project=wandb_config['project'])
                cls.args.use_wandb = True
            except Exception as e:
                raise Exception(f"Error opening WandB config file: {cls.args.config}") from e

        # visualize config
        if getattr(cls.args, "command", None) == "visualize" or getattr(cls.args, "command", None) == "train":
            # open yaml config file
            if getattr(cls.args, "verbose", False):
                print(f"Opening config file {cls.args.config}")

            try:
                with open(cls.args.config, 'r') as file:
                    CTX = yaml.safe_load(file)
                cls.args.config = CTX
            except Exception as e:
                raise Exception(f"Error opening config file: {cls.args.config}") from e
