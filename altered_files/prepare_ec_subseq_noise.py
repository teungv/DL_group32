"""
Prepare data for a subset of an Event Camera Dataset sequence
- Undistort images and events
- Create time surfaces
- Create an output directory with undistorted images, undistorted event txt, and time surfaces
"""
import os
import shutil
from glob import glob
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
from fire import Fire
from matplotlib import pyplot as plt
from pandas import read_csv
from tqdm import tqdm
import sys

# Get the absolute path of the script
script_path = os.path.abspath(__file__)

# Get the absolute path of the 'deep_ev_tracker' directory
deep_ev_tracker_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

# Add the 'deep_ev_tracker' directory to sys.path
if deep_ev_tracker_dir not in sys.path:
    sys.path.append(deep_ev_tracker_dir)

from utils.utils import blosc_opts


def prepare_data(root_dir, sequence_name, start_idx, end_idx):
    sequence_dir = Path(root_dir) / sequence_name
    if not sequence_dir.exists():
        print(f"Sequence directory does not exist for {sequence_name}")
        exit()

    # Read calib
    calib_data = np.genfromtxt(str(sequence_dir / "calib.txt"))
    camera_matrix = calib_data[:4]
    distortion_coeffs = calib_data[4:]
    camera_matrix = np.array(
        [
            [camera_matrix[0], 0, camera_matrix[2]],
            [0, camera_matrix[1], camera_matrix[3]],
            [0, 0, 1],
        ]
    )
    print("Calibration loaded")

    # Create output directory
    subseq_dir = Path(root_dir) / f"{sequence_name}_{start_idx}_{end_idx}"
    subseq_dir.mkdir(exist_ok=True)

    # Undistort images
    images_dir = sequence_dir / "images_corrected"
    if not images_dir.exists():
        images_dir.mkdir()
        for img_idx, img_path in enumerate(
            tqdm(
                sorted(glob(os.path.join(str(sequence_dir / "images" / "*.png")))),
                desc="Undistorting images...",
            )
        ):
            img = cv2.imread(img_path)
            img = cv2.undistort(
                img, cameraMatrix=camera_matrix, distCoeffs=distortion_coeffs
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filename = f"frame_{str(img_idx).zfill(8)}.png"
            cv2.imwrite(os.path.join(str(images_dir / filename)), img)
    img_tmp = cv2.imread(str(images_dir / "frame_00000000.png"))
    H_img, W_img = img_tmp.shape[:2]

    # Remove first entry in image timestamps
    image_timestamps = np.genfromtxt(str(sequence_dir / "images.txt"), usecols=[0])
    image_timestamps = image_timestamps[1:]
    np.savetxt(str(sequence_dir / "images.txt"), image_timestamps)

    # Undistort events
    events_corrected_path = sequence_dir / "events_corrected.txt"
    if not events_corrected_path.exists():
        events = read_csv(
            str(sequence_dir / "events.txt"), header=None, delimiter=" "
        ).to_numpy()
        print("Raw events loaded")

        events[:, 1:3] = cv2.undistortPoints(
            events[:, 1:3].reshape((-1, 1, 2)),
            camera_matrix,
            distortion_coeffs,
            P=camera_matrix,
        ).reshape(
            (-1, 2),
        )
        events[:, 1:3] = np.rint(events[:, 1:3])

        inbounds_mask = np.logical_and(events[:, 1] >= 0, events[:, 1] < W_img)
        inbounds_mask = np.logical_and(inbounds_mask, events[:, 2] >= 0)
        inbounds_mask = np.logical_and(inbounds_mask, events[:, 2] < H_img)
        events = events[inbounds_mask, :]

        print("Events undistorted")
        np.savetxt(events_corrected_path, events, ["%.9f", "%i", "%i", "%i"])
    else:
        events = read_csv(
            str(events_corrected_path), header=None, delimiter=" "
        ).to_numpy()
    t_events = events[:, 0]

    subseq_images_dir = subseq_dir / "images_corrected"
    if not subseq_images_dir.exists():
        subseq_images_dir.mkdir()

    for i in range(start_idx, end_idx + 1):
        shutil.copy(
            str(images_dir / f"frame_{str(i).zfill(8)}.png"),
            str(subseq_images_dir / f"frame_{str(i-start_idx).zfill(8)}.png"),
        )

    # Get image dimensions
    IMG_H, IMG_W = cv2.imread(
        str(images_dir / "frame_00000001.png"), cv2.IMREAD_GRAYSCALE
    ).shape

    # Read image timestamps
    image_timestamps = np.genfromtxt(sequence_dir / "images.txt", usecols=[0])
    image_timestamps = image_timestamps[start_idx : end_idx + 1]
    np.savetxt(str(subseq_dir / "images.txt"), image_timestamps)
    print(
        f"Image timestamps are in range [{image_timestamps[0]}, {image_timestamps[-1]}]"
    )
    print(f"Event timestamps are in range [{t_events.min()}, {t_events.max()}]")

    # Copy calib and poses
    shutil.copy(str(sequence_dir / "calib.txt"), str(subseq_dir / "calib.txt"))
    shutil.copy(
        str(sequence_dir / "groundtruth.txt"), str(subseq_dir / "groundtruth.txt")
    )

    # Generate debug frames
    debug_dir = sequence_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)
    n_frames_debug = 0
    dt = 0.005
    for i in range(n_frames_debug): # this is done once? 
        # Events
        t1 = image_timestamps[i] # 
        t0 = t1 - dt
        time_mask = np.logical_and(events[:, 0] >= t0, events[:, 0] < t1)
        events_slice = events[time_mask, :]

        on_mask = events_slice[:, 3] == 1
        off_mask = events_slice[:, 3] == 0
        events_slice_on = events_slice[on_mask, :]
        events_slice_off = events_slice[off_mask, :]

        # Image
        img = cv2.imread(
            str(images_dir / f"frame_{str(i).zfill(8)}.png"), cv2.IMREAD_GRAYSCALE
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img, cmap="gray")
        ax.scatter(events_slice_on[:, 1], events_slice_on[:, 2], s=5, c="green")
        ax.scatter(events_slice_off[:, 1], events_slice_off[:, 2], s=5, c="red")
        plt.show()
        fig.savefig(str(debug_dir / f"frame_{str(i).zfill(8)}.png"))
        fig.close()

    # Add noise
    noise_level = 0.1 # 0.1 = 10% of pixels flipped
    noise_mode = "dropped" # choose between "flipped" and "dropped"

    assert noise_level >= 0 and noise_level <= 1, f'Noise level should be between 0 and 1, got {noise_level}'
    assert noise_mode in ["flipped", "dropped"], f'Noise mode should be "flipped" or "dropped", got {noise_mode}'

    # Determine the number of pixels to flip based on the noise level
    num_pixels_to_flip = int(noise_level * IMG_H * IMG_W)
    

    # Generate time surfaces
    for dt in [0.01]:
        for n_bins in [5]:
            dt_bin = dt / n_bins
            output_ts_dir = (
                subseq_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}" / str(noise_level)
            )
            if not output_ts_dir.exists():
                output_ts_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = subseq_dir / f"debug_events_{n_bins}" / str(noise_level)
            debug_dir.mkdir(exist_ok=True, parents=True)
            for i, t1 in tqdm(
                enumerate(
                    np.arange(image_timestamps[0], image_timestamps[-1] + dt, dt)
                ),
                total=int((image_timestamps[-1] - image_timestamps[0]) / dt),
                desc="Generating time surfaces...",
            ): # for every time step for this image with steps of size dt
                output_ts_path = (
                    output_ts_dir / f"{str(int(i * (dt * 1e6))).zfill(7)}.h5"
                )
                #if output_ts_path.exists():
                #    continue

                # for every time bin, we have two channels: one for positive events and one for negative events    
                time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.float64)
                # t0 is the start of the time bin
                t0 = t1 - dt

                # iterate over bins
                for i_bin in range(n_bins):
                    t0_bin = t0 + i_bin * dt_bin # within the bin we also iterate and have a time step
                    t1_bin = t0_bin + dt_bin # end of the bin

                    time_mask = np.logical_and(
                        events[:, 0] >= t0_bin, events[:, 0] < t1_bin
                    ) # get the events that are in the bin
                    events_slice = events[time_mask, :] # use the mask to get the proper events

                    for i in range(events_slice.shape[0]): # for every event in the bin
                        if (
                            0 <= events_slice[i, 2] < IMG_H # check if the event is within the image
                            and 0 <= events_slice[i, 1] < IMG_W 
                        ):
                            time_surface[
                                int(events_slice[i, 2]), # row
                                int(events_slice[i, 1]), # column
                                2 * i_bin + int(events_slice[i, 3]), # channel
                            ] = (
                                events_slice[i, 0] - t0 # time of the event
                            )
                    
                    if noise_mode == "flipped":
                        for bin_val in range(2):
                            # Generate random indices to flip
                            all_indices = np.arange(IMG_H * IMG_W)
                            flip_indices = np.random.choice(all_indices, num_pixels_to_flip, replace=False)

                            # Convert the flattened index to row and column indices
                            row_index = flip_indices // IMG_W # in the first 240 pixels -> 1 -> row 1
                            col_index = flip_indices % IMG_W # in the 490 / 240 = 10 is the 10th column

                            # We are dealing with time surface. Therefore, if the index value is non zero, we sent it to 0, otherwise, we set it to a random correct value
                            for row, col in zip(row_index, col_index):
                                element = time_surface[row, col, i_bin + bin_val] # get the value of the event
                                if element != 0: # if the event is not zero, we set it to zero
                                    time_surface[row, col, i_bin + bin_val] = 0.0
                                else: # if the event is zero, we set it to a random time, so a random event
                                    noise_event_time = round(np.random.uniform(t0_bin, t1_bin), 6)
                                    time_surface[row, col, i_bin + bin_val] = noise_event_time - t0

                    elif noise_mode == "dropped":
                        for bin_val in range(2):

                            event_indices = np.array([
                            (row, col) for row in range(IMG_H) for col in range(IMG_W)
                            if time_surface[row, col, i_bin + bin_val] != 0
                                                    ])
                            number_of_events_to_drop = int(noise_level * len(event_indices))
                            drop_indices = np.random.choice(len(event_indices), number_of_events_to_drop, replace=False)

                            for index in drop_indices:
                                row, col = event_indices[index]
                                time_surface[row, col, i_bin + bin_val] = 0

                            # event_indices_post_drop = np.array([
                            # (row, col) for row in range(IMG_H) for col in range(IMG_W)
                            # if time_surface[row, col, i_bin + bin_val] != 0
                            #                         ])
                            
                            # print(f'amount of events post drop: {len(event_indices_post_drop)}')

                time_surface = np.divide(time_surface, dt)

                with h5py.File(output_ts_path, "w") as h5f_out:
                    h5f_out.create_dataset(
                        "time_surface",
                        data=time_surface,
                        shape=time_surface.shape,
                        dtype=np.float32,
                        **blosc_opts(complevel=1, shuffle="byte"),
                    )

                # Visualize
                debug_event_frame = ((time_surface[:, :, 0] > 0) * 255).astype(np.uint8)
                cv2.imwrite(
                    str(debug_dir / f"{str(int(i * dt * 1e6)).zfill(7)}.png"),
                    debug_event_frame,
                )


if __name__ == "__main__":
    #Fire(prepare_data)

    # sequence_root = 'C://Users//Administrateur//Documents//TUDelft//scripts//CS4240//Project//ec_subseq/test'
    sequence_root = '/home/teunv/Documents/DL/Project/ec_subseq/clean_data'
    list_to_generate = [('boxes_rotation', 198, 278), ('boxes_translation', 330, 410), ('shapes_6dof', 485, 565), ('shapes_rotation', 165, 245), ('shapes_translation', 8, 88)]
    # list_to_generate = [('boxes_rotation', 198, 278)]
    for (sequence_name, start_idx, end_idx) in list_to_generate:
        root_dir = sequence_root
        prepare_data(root_dir=root_dir, sequence_name=sequence_name, start_idx=start_idx, end_idx=end_idx)
