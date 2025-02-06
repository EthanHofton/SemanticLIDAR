import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from args.args import Args

def check_paths():
    if not os.path.exists(Args.args.lidar):
        raise Exception(f"Error: Lidar data path {args.lidar} does not exist.")
    if not os.path.exists(Args.args.calib):
        raise Exception(f"Error: Calibration data path {args.calib} does not exist.")
    if not os.path.exists(Args.args.labels):
        raise Exception(f"Error: Labels data path {args.labels} does not exist.")

def count_dirs(path):
    return sum(1 for item in path.iterdir() if item.is_dir())

def copy_data(src, dest, rsync, msg):
    print(msg)
    if rsync:
        subprocess.run(['rsync', '-a', '--no-relative', '--inplace', src, dest])
    else:
        subprocess.run(['cp', '-r', src, dest])

def sequence(i, base_lidar, base_calib, base_labels, base_output):
    folder_name = f"{i:02}"
    print(f"Creating folder for sequnce {folder_name}")

    # Get data sequence
    lidar_sequence = base_lidar / folder_name
    calib_sequence = base_calib / folder_name
    label_sequence = base_labels / folder_name

    if not lidar_sequence.is_dir():
        raise Exception(f"Lidar: sequence {folder_name} not found")
    if not calib_sequence.is_dir():
        raise Exception(f"Calib: sequence {folder_name} not found")
    if not label_sequence.is_dir():
        raise Exception(f"Label: sequence {folder_name} not found")

    # Create Data sequence
    output_sequence = base_output / folder_name
    output_sequence.mkdir(exist_ok=False)

    # Copy calib data:
    copy_data(str(calib_sequence) + '/', str(output_sequence), Args.args.rsync, 'Copying calib data')
    copy_data(str(label_sequence) + '/', str(output_sequence), Args.args.rsync, 'Copying label data')
    copy_data(str(lidar_sequence) + '/', str(output_sequence), Args.args.rsync, 'Copying LiDAR data')


def make_dataset():
    # check paths exist
    check_paths()

    # Create the output folder
    os.makedirs(Args.args.output, exist_ok=False)
    base_output = Path(Args.args.output)

    base_lidar = Path(Args.args.lidar)
    base_calib = Path(Args.args.calib)
    base_labels = Path(Args.args.labels)

    # find the sequences dir
    if (base_lidar / "sequences").is_dir():
        base_lidar = base_lidar / "sequences"
    else:
        raise Exception("LiDAR: sequnces not found")

    if (base_calib / "sequences").is_dir():
        base_calib = base_calib / "sequences"
    else:
        raise Exception("Calib: sequnces not found")

    if (base_labels / "sequences").is_dir():
        base_labels =  base_labels / "sequences"
    else:
        raise Exception("Labels: sequnces not found")

    # create sequences for output
    base_output = base_output / "sequences"
    base_output.mkdir(exist_ok=False)

    # loop through sequences
    assert count_dirs(base_lidar) == count_dirs(base_calib) == count_dirs(base_labels)
    num_sequences = count_dirs(base_lidar)

    if Args.args.verbose:
        print(f"{num_sequences} sequences found")

    with ThreadPoolExecutor() as executor:
        futures = []

        # submit each sequence to a thread
        for i in range(0, num_sequences):
            futures.append(executor.submit(sequence, i, base_lidar, base_calib, base_labels, base_output))
        
        # wait for all threads to rejoin
        for future in futures:
            future.result()
