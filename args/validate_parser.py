def add_validate_parser(subparsers):
    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('-d',
                              '--dataset',
                              type=str,
                              help='Path to dataset',
                              required=True)
    validate_parser.add_argument('-c',
                              '--ds_config',
                              type=str,
                              default='configs/semantic-kitti.yaml',
                              help='Config file for KITTI dataset')
    validate_parser.add_argument('--verbose',
                              action='store_true',
                              help='Give verbose output')
    validate_parser.add_argument('--device',
                              type=str,
                              help='Training device to use. CPU or MPS',
                              default='cpu')
    validate_parser.add_argument('--model',
                              help='load a model from a checkpoint to resume training',
                              default=None,
                              type=str)
    validate_parser.add_argument('--downsample',
                                 help='downsample the point cloud',
                                 action='store_true')
    validate_parser.add_argument('--run_config',
                                 help='The run config',
                                 type=str,
                                 default='configs/val_run_config.yaml')
    validate_parser.set_defaults(command='validate')

