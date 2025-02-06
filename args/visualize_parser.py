def add_visualize_parser(subparsers):
    visualize_parser = subparsers.add_parser('visualize')
    visualize_parser.add_argument('-d',
                                  '--dataset',
                                  type=str,
                                  help='Path to dataset',
                                  required=True)
    visualize_parser.add_argument('-p',
                                  '--predictions',
                                  type=str,
                                  help='Visualize custom labels',
                                  default=None,
                                  required=False)
    visualize_parser.add_argument('--offset',
                                  type=int,
                                  help='Offset from start of sqeunce',
                                  default=0,
                                  required=False)
    visualize_parser.add_argument('-s',
                                  '--sequence',
                                  type=str,
                                  help='Sequence folder to read from',
                                  default='00',
                                  required=False)
    visualize_parser.add_argument('-c',
                                  '--config',
                                  type=str,
                                  help='Label config file',
                                  default='configs/semantic-kitti.yaml',
                                  required=False)
    visualize_parser.add_argument('--verbose',
                                  action='store_true',
                                  help='Give verbose output to the terminal')
    visualize_parser.set_defaults(command='visualize')

