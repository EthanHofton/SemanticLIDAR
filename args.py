import yaml

class Args():
    args = {}

    @classmethod
    def init(cls, args):
        cls.args = args

        # visualize config
        if getattr(cls.args, "command", None) == "visualize":
            # open yaml config file
            if getattr(cls.args, "verbose", False):
                print(f"Opening config file {cls.args.config}")

            try:
                with open(cls.args.config, 'r') as file:
                    CTX = yaml.safe_load(file)
                cls.args.config = CTX
            except Exception as e:
                raise Exception(f"Error opening config file: {cls._config.config}") from e
