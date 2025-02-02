from util.auxiliary.laserscan import LaserScan, SemLaserScan
import open3d as o3d
import os
import yaml

def visualize(args):
    # open config file
    try:
        if args.verbose:
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
        if args.verbose:
            print(f"Using sequence from {scan_paths}")
    else:
        print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
        raise Exception(f"Failed to open sequence folder {scan_paths}")

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
        if args.verbose:
            print(f"Using labels from {label_paths}")
    else:
        print(f"Labels folder {label_paths} doesn't exist! Exiting...")
        raise Exception(f"Failed to open {label_paths}")

    # create a scan
    color_dict = CFG["color_map"]

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

    if args.verbose:
        print("Loaded point cloud")
        print("Visualizing... press h for help")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
