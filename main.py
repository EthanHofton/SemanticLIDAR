from args.parse_args import parse_args
from args.args import Args

from util.visualize import visualize
from util.make_dataset import make_dataset

from training.train import train

if __name__ == "__main__":
    args, print_help = parse_args()
    Args.init(args)

    if Args.args.command == 'visualize':
        visualize()
    elif Args.args.command == 'train':
        train()
    elif Args.args.command == 'make_dataset':
        make_dataset()
    else:
        print_help()

