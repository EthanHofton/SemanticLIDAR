def add_make_dataset_parser(subparsers):
    make_dataset_parser = subparsers.add_parser('make_dataset')
    make_dataset_parser.add_argument(
        '--lidar', 
        type=str, 
        required=True, 
        help='Path to the lidar data'
    )
    make_dataset_parser.add_argument(
        '--calib', 
        type=str, 
        required=True, 
        help='Path to the calibration data'
    )
    make_dataset_parser.add_argument(
        '--labels', 
        type=str, 
        required=True, 
        help='Path to the labels data'
    )
    make_dataset_parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='Path to save the output dataset'
    )
    make_dataset_parser.add_argument(
        '--rsync',
        action='store_true',
        help='Use rsync for copying files (default: cp)'
    )
    make_dataset_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Give verbose output'
    )
    make_dataset_parser.set_defaults(command='make_dataset')
