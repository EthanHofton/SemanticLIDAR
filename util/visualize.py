from util.auxiliary.laserscan import LaserScan, SemLaserScan
import open3d as o3d
import os
import yaml
from args.args import Args

def visualize():
    # fix sequence name
    Args.args.sequence = '{0:02d}'.format(int(Args.args.sequence))
    scan_paths = os.path.join(Args.args.dataset, "sequences", 
                              Args.args.sequence, "velodyne")
    # check scan path is path
    if os.path.isdir(scan_paths):
        if Args.args.verbose:
            print(f"Using sequence from {scan_paths}")
    else:
        print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
        raise Exception(f"Failed to open sequence folder {scan_paths}")

    # construct scan directory
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # load label paths
    if Args.args.predictions is not None:
        label_paths = os.path.join(Args.args.predictions, "sequences",
                                   Args.args.sequence, "predictions")
    else:
        label_paths = os.path.join(Args.args.dataset, "sequences",
                                   Args.args.sequence, "labels")

    # construct the label names
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    # check label paths exist
    if os.path.isdir(label_paths):
        if Args.args.verbose:
            print(f"Using labels from {label_paths}")
    else:
        print(f"Labels folder {label_paths} doesn't exist! Exiting...")
        raise Exception(f"Failed to open {label_paths}")

    # create a scan
    color_dict = Args.args.config["color_map"]

    scan = SemLaserScan(color_dict, project=True)
    scan.open_scan(scan_names[0])
    scan.open_label(label_names[0])
    scan.colorize()

    points = scan.points
    sem_colors = scan.sem_label_color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Use semantic colours for the point cloud
    pcd.colors = o3d.utility.Vector3dVector(sem_colors)

    if Args.args.verbose:
        print("Loaded point cloud")
        print("Visualizing... press h for help")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
