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

    validate_parser.add_argument('--per_class',
                                 help='View the per-class metrics or total average',
                                 action='store_true')
    validate_parser.add_argument('--confusion_matrix',
                                 help='Save a confustion matrix',
                                 action='store_true')
    validate_parser.add_argument('--save',
                                 help='Save the metrics to a file',
                                 action='store_true')
    validate_parser.add_argument('--view',
                                 help='View a plot of the metrics',
                                 action='store_true')
    validate_parser.set_defaults(command='validate')

