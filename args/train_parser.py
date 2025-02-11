def add_train_parser(subparsers):
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-d',
                              '--dataset',
                              type=str,
                              help='Path to dataset',
                              required=True)
    train_parser.add_argument('-c',
                              '--ds_config',
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
    train_parser.add_argument('--checkpoint',
                              help='frequences to checkpoint models per epoch. 0 for none',
                              default=0,
                              type=int)
    train_parser.add_argument('--from_checkpoint',
                              help='load a model from a checkpoint to resume training',
                              default=None,
                              type=str)
    train_parser.add_argument('--wandb',
                              help='Enable wandb logging, provide wandb config file',
                              action='store_true')
    train_parser.add_argument('--run_config',
                              help='run_config yaml file',
                              type=str,
                              default='configs/run_config.yaml')
    train_parser.set_defaults(command='train')
