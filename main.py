import argparse
import os
import yaml
from util.auxiliary.laserscan import LaserScan, SemLaserScan
import open3d as o3d

def visualize(args):
    # open config file
    # open config file
    try:
        print("Opening config file %s" % args.config)
        CFG = yaml.safe_load(open(args.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # fix sequence name
    args.sequence = '{0:02d}'.format(int(args.sequence))
    scan_paths = os.path.join(args.dataset, "sequences", 
                              args.sequence, "velodyne")
    # check scan path is path
    if os.path.isdir(scan_paths):
        print(f"Sequence folder {scan_paths} exists! Using sequence from {scan_paths}")
    else:
        print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
        quit()

    # construct scan directory
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # load label paths
    if args.predictions is not None:
        label_paths = os.path.join(args.predictions, "sequences",
                                   args.sequence, "predictions")
    else:
        label_paths = os.path.join(args.dataset, "sequences",
                                   args.sequence, "labels")

    # construct the label names
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    # check label paths exist
    if os.path.isdir(label_paths):
        print(f"Labels folder {label_paths} exists! Using labels from {label_paths}")
    else:
        print(f"Labels folder {label_paths} doesn't exist! Exiting...")
        quit()

    # create a scan
    color_dict = CFG["color_map"]

    scan = SemLaserScan(color_dict, project=True)
    scan.open_scan(scan_names[0])
    scan.open_label(label_names[0])
    scan.colorize()

    # plt.figure(figsize=(15, 10))
    #
    # # Range image
    # plt.subplot(1, 2, 1)
    # plt.title("Range Image")
    # plt.imshow(scan.proj_range, cmap='viridis')
    # plt.colorbar(label="Range (m)")
    #
    # # Semantic label image
    # plt.subplot(1, 2, 2)
    # plt.title("Semantic Labels")
    # plt.imshow(scan.proj_sem_color)
    # plt.colorbar(label="Semantic Colour")
    #
    # plt.show()
    # Create a figure for the 3D plot
    points = scan.points
    sem_colors = scan.sem_label_color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Use semantic colours for the point cloud
    pcd.colors = o3d.utility.Vector3dVector(sem_colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # Save the point cloud to a file
    o3d.io.write_point_cloud("point_cloud_visualization.ply", pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SemanticKITTI solution',
                                     description='Training, Testing and util tools for SemanticKITTI solution')

    subparsers = parser.add_subparsers(title='subcommands',
                                       description='The subcommands of the program',
                                       help='additional help',
                                       required=True)

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
    visualize_parser.add_argument('--output',
                                  type=str,
                                  help='Output file for visualized lidar cloud',
                                  default=None,
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
    visualize_parser.set_defaults(command='visualize')

    args = parser.parse_args()

    if args.command == 'visualize':
        visualize(args)
    else:
        parser.print_help()

