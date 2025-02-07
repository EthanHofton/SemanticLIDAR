def add_train_parser(subparsers):
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-d',
                              '--dataset',
                              type=str,
                              help='Path to dataset',
                              required=True)
    train_parser.add_argument('-c',
                              '--config',
                              type=str,
                              default='configs/semantic-kitti.yaml',
                              help='Config file for KITTI dataset')
    train_parser.add_argument('--verbose',
                              action='store_true',
                              help='Give verbose output')
    train_parser.add_argument('--device',
                              type=str,
                              help='Training device to use. CPU or MPS',
                              default='cpu')
    train_parser.add_argument('--validate',
                              help='Validate each epoch using a validation set',
                              action='store_true')
    train_parser.set_defaults(command='train')
