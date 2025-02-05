from util.parseargs import parse_args
from util.visualize import visualize
from args import Args

if __name__ == "__main__":
    args = parse_args()
    Args.init(args)

    if Args.args.command == 'visualize':
        visualize()
    else:
        parser.print_help()

