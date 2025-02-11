import yaml
import wandb
from util.get_device import get_device
from util.run_config import RunConfig, WandBRunConfig

class Args():
    args = {}

    @classmethod
    def init(cls, args):
        cls.args = args

        if getattr(cls.args, "device", None):
            # check device is avaliable on OS
            cls.args.device = get_device(cls.args.device)

        if getattr(cls.args, "run_config", None):
            try:
                with open(cls.args.run_config, 'r') as file:
                    config = yaml.safe_load(file)
                if getattr(cls.args, "verbose", False):
                    print(f"Loaded run config: {cls.args.run_config}")

                if getattr(cls.args, "wandb", False):
                    wandb.init(config=config)
                    cls.run_config = WandBRunConfig()
                else:
                    cls.run_config = RunConfig(config)
            except Exception as e:
                raise Exception(f"Error opening run config file: {cls.args.run_config}")

        # visualize config
        if getattr(cls.args, "command", None) == "visualize" or getattr(cls.args, "command", None) == "train" or getattr(cls.args, "command", None) == "validate":
            # open yaml config file
            if getattr(cls.args, "verbose", False):
                print(f"Opening dataset config file {cls.args.ds_config}")

            try:
                with open(cls.args.ds_config, 'r') as file:
                    CTX = yaml.safe_load(file)
                cls.args.ds_config = CTX
            except Exception as e:
                raise Exception(f"Error opening config file: {cls.args.ds_config}") from eargs
