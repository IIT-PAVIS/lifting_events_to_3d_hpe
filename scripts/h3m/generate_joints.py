import argparse
import os

import event_library as el
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from utils import (
    constant_count_joint_generator,
    joint_generator,
    timestamps_generator,
    voxel_grid_joint_generator,
)

from experimenting.dataset import HumanCore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Accumulates events to an event-frame."
    )
    parser.add_argument("--event_files", nargs="+", help="file(s) to convert to output")
    parser.add_argument(
        "--joints_file",
        type=str,
        help="file of .npz joints containing joints data. Generate it using `prepare_data_h3m`",
    )
    parser.add_argument("--output_base_dir", type=str, help="output_dir")
    parser.add_argument(
        "--representation",
        type=str,
        default='constant-count',
        help="representation to use for generating events frames. Choose between [constant-count, voxel-grid]",
    )
    parser.add_argument(
        '--generate-joints',
        action='store_true',
        help='If set, generate also joints labels synchronized with event frames',
    )
    parser.add_argument(
        "--num_events", type=int, default=30000, help="num events to accumulate"
    )

    args = parser.parse_args()
    return args


def _get_multicam_events(event_files, idx, n_cameras=4):
    events = []
    for offset_id in range(0, n_cameras):
        events.append(el.utils.load_from_file(event_files[idx + offset_id]))

    events = [
        np.concatenate(
            [events[index], index * np.ones((len(events[index]), 1))], axis=1
        )
        for index in range(n_cameras)
    ]

    events = np.concatenate(events)
    sort_index = np.argsort(events[:, 2])
    events = events[sort_index]
    return events


def _generate_joints(
    events: np.array,
    joints: np.array,
    subject: str,
    action: str,
    output_joint_path: str,
) -> np.array:

    gt_generator = joint_generator(events, joints, num_events)
    joints = []
    for joint_frame in gt_generator:
        joints.append(joint_frame)
    return np.stack(joints)


if __name__ == '__main__':
    args = parse_args()
    event_files = args.event_files
    joints_file = args.joints_file
    num_events = args.num_events
    data = HumanCore.get_pose_data(joints_file)
    output_base_dir = args.output_base_dir
    hw_info = el.utils.get_hw_property('dvs')
    n_cameras = 4  # Number of parallel cameras
    switch = {
        'constant-count': constant_count_joint_generator,
        'voxel-grid': voxel_grid_joint_generator,
    }
    os.makedirs(output_base_dir)
    output_joint_path = os.path.join(output_base_dir, "3d_joints")

    joint_gt = {f"S{s:01d}": {} for s in range(1, 12)}
    timestamps = {f"S{s:01d}": {} for s in range(1, 12)}

    cam_index_to_id_map = dict(
        zip(HumanCore.CAMS_ID_MAP.values(), HumanCore.CAMS_ID_MAP.keys())
    )
    representation_generator = switch[args.representation]
    # representation_generator = timestamps_generator

    def _fun(idx):
        info = HumanCore.get_frame_info(event_files[idx])
        action = info['action']
        action = action.replace("TakingPhoto", "Photo").replace("WalkingDog", "WalkDog")

        if info['subject'] == 11 and action == "Directions":
            print(f"Discard {info}")
            return

        if "_ALL" in action:
            print(f"Discard {info}")
            return

        output_dir = os.path.join(
            output_base_dir, f"S{info['subject']:01d}", f"{action}" + ".{}"
        )

        timestamps[f"S{info['subject']:01d}"][action] = []
        joints = data[info['subject']][action]['positions']

        events = _get_multicam_events(event_files, idx, n_cameras)
        frame_generator = representation_generator(
            events, joints, num_events, hw_info.size
        )
        for ind_frame, events_per_cam in enumerate(frame_generator):
            event_frame_per_cams, timestamp = events_per_cam
            for id_camera in range(n_cameras):
                cam = cam_index_to_id_map[id_camera]
                os.makedirs(output_dir.format(cam), exist_ok=True)
                np.save(
                    os.path.join(output_dir.format(cam), f"frame{ind_frame:07d}.npy"),
                    event_frame_per_cams[id_camera],
                )

        if args.generate_joints:
            joint_gt[f"S{info['subject']:01d}"][action] = _generate_joints(
                events, joints
            )

    thread_map(_fun, list(range(0, len(event_files), n_cameras)), max_workers=16)

    save_params = {"timestamps": timestamps}
    if args.generate_joints:
        save_params["positions_3d"] = joint_gt

    np.savez_compressed(output_joint_path, **save_params)
