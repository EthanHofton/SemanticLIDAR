from util.parseargs import parse_args
from util.visualize import visualize

if __name__ == "__main__":
    args = parse_args()

    if args.command == 'visualize':
        visualize(args)
    else:
        parser.print_help()

