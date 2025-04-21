import math
import numpy as np
import torch
import os
import glob
from datetime import datetime
import argparse

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor


from dataclasses import dataclass, field
from telemoma.utils.general_utils import AttrDict

import argparse

from multiprocessing import (
    Array,
    Process,
    shared_memory,
    Queue,
    Manager,
    Event,
    Semaphore,
)

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.objects import PrimitiveObject
from omnigibson.envs import DataCollectionWrapper
from omnigibson.utils.ui_utils import KeyboardEventHandler
import omnigibson.lazy as lazy

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = False  # TODO: turn this on for speed

CAMERA_HEIGHT = 720
CAMERA_WIDTH = 1280

THUMB_TIP = 4
INDEX_TIP = 9
MIDDLE_TIP = 14
RING_TIP = 19
PINKY_TIP = 24
PALM_CENTER = 0

RECORDING_PATH = None


def setup_recording_path(base_path, user_name):
    """
    Set up the recording path for the user and determine the next trial number.

    Args:
        base_path (str): Base directory for recordings
        user_name (str): Name of the user

    Returns:
        str: Full path for recording file
    """
    # Create user directory if it doesn't exist
    user_dir = os.path.join(base_path, user_name)
    os.makedirs(user_dir, exist_ok=True)

    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find existing trial files to determine next trial number
    existing_files = glob.glob(os.path.join(user_dir, f"vision_pro_{user_name}_trial*"))
    trial_numbers = [0]  # Start with 0 in case no files exist

    for file in existing_files:
        try:
            # Extract trial number from filename
            trial_str = file.split("trial")[-1].split("_")[0]
            trial_numbers.append(int(trial_str))
        except (ValueError, IndexError):
            continue

    next_trial = max(trial_numbers) + 1

    # Create filename
    filename = f"vision_pro_{user_name}_trial{next_trial}_{current_time}.hdf5"
    full_path = os.path.join(user_dir, filename)

    return full_path


def create_camera(relative_prim_path, scene):
    camera = VisionSensor(
        relative_prim_path=relative_prim_path,
        name=relative_prim_path.split("/")[
            -1
        ],  # Assume name is the lowest-level name in the prim_path
        modalities="rgb",
        image_height=CAMERA_HEIGHT,
        image_width=CAMERA_WIDTH,
    )
    camera.load(scene)

    # We update its clipping range and focal length so we get a good FOV and so that it doesn't clip
    # nearby objects (default min is 1 m)
    camera.clipping_range = [0.001, 10000000.0]
    camera.focal_length = 17.0

    camera.initialize()

    return camera


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))


@dataclass
class TeleopAction(AttrDict):
    left: torch.Tensor = field(
        default_factory=lambda: torch.cat((torch.zeros(6), torch.ones(1)))
    )
    right: torch.Tensor = field(
        default_factory=lambda: torch.cat((torch.zeros(6), torch.ones(1)))
    )
    base: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    torso: float = field(
        default_factory=lambda: torch.zeros(6)
    )  # Using IK controller (pose_delta_ori mode) to control torso; 6DOF (dx,dy,dz,dax,day,daz) control over pose


class OGTeleop:
    def __init__(self):
        self.resolution = (CAMERA_HEIGHT, CAMERA_WIDTH)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.img_shape) * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, image_queue, toggle_streaming
        )
        # self.processor = OGPreprocessor()
        self.processor = VuerPreprocessor()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_fingers, right_fingers = (
            self.processor.process(self.tv)
        )

        head_rotation = torch.tensor(head_mat[:3, :3], dtype=torch.float32)
        head_translation = torch.tensor(head_mat[:3, 3], dtype=torch.float32)

        # if abs(T.mat2euler(head_rotation)[2]) > math.pi / 2.0:
        #     breakpoint()

        left_pose = (
            torch.tensor(left_wrist_mat[:3, 3], dtype=torch.float32),
            T.mat2quat(torch.tensor(left_wrist_mat[:3, :3], dtype=torch.float32)),
        )
        right_pose = (
            torch.tensor(right_wrist_mat[:3, 3], dtype=torch.float32),
            T.mat2quat(torch.tensor(right_wrist_mat[:3, :3], dtype=torch.float32)),
        )

        left_is_grasping = self.is_grasping(left_fingers)
        right_is_grasping = self.is_grasping(right_fingers)

        return (
            head_rotation,
            head_translation,
            left_pose,
            right_pose,
            left_is_grasping,
            right_is_grasping,
        )

    def is_grasping(self, hand_landmarks, distance_threshold=0.1):
        """
        Detect if the hand is making a grasping gesture

        Args:
            hand_landmarks: numpy array of shape (25, 3) containing 3D hand landmarks
            distance_threshold: maximum distance between thumb and fingers to be considered grasping

        Returns:
            bool: True if grasping gesture is detected, False otherwise
        """
        # Get fingertip positions
        thumb = hand_landmarks[THUMB_TIP]
        index = hand_landmarks[INDEX_TIP]
        middle = hand_landmarks[MIDDLE_TIP]
        ring = hand_landmarks[RING_TIP]
        pinky = hand_landmarks[PINKY_TIP]
        palm = hand_landmarks[PALM_CENTER]

        # Calculate distances between thumb and other fingertips
        distances = {
            "thumb_index": calculate_distance(thumb, index),
            "thumb_middle": calculate_distance(thumb, middle),
            "thumb_ring": calculate_distance(thumb, ring),
            "thumb_pinky": calculate_distance(thumb, pinky),
        }

        # Calculate average finger spread (distance from palm to fingertips)
        finger_spread = np.mean(
            [
                calculate_distance(palm, index),
                calculate_distance(palm, middle),
                calculate_distance(palm, ring),
                calculate_distance(palm, pinky),
            ]
        )

        # Conditions for grasp detection:
        # 1. At least 3 fingers are close to the thumb
        fingers_grasping = sum(1 for d in distances.values() if d < distance_threshold)
        is_close_grasp = fingers_grasping >= 3

        # 2. Fingers should be curled (closer to palm than in open hand)
        # Assuming typical open hand finger spread is around 0.15-0.2 units
        is_fingers_curled = finger_spread < 0.15

        # metrics = {
        #     "distances": distances,
        #     "finger_spread": finger_spread,
        #     "fingers_grasping": fingers_grasping,
        #     "is_close_grasp": is_close_grasp,
        #     "is_fingers_curled": is_fingers_curled,
        # }

        # Both conditions should be met for a grasp
        return is_close_grasp and is_fingers_curled


class Sim:
    def __init__(self, debug_mode=False, use_robot_camera=False, base_mode="position"):

        motor_type = base_mode if base_mode == "velocity" else "position"
        # scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}
        # scene_cfg = {"type": "Scene"}
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Pomaria_1_int",
            "load_task_relevant_only": False,
        }
        task_cfg = {
            "type": "BehaviorTask",
            "activity_name": "clean_your_house_after_a_wild_party",
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
        }
        robot_cfg = {
            "type": "R1",
            "position": [-3.3066, -1.5145, 0.0422],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "obs_modalities": ["rgb"],
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "mode": "absolute_pose",
                    "command_input_limits": None,
                    "command_output_limits": None,
                    # "use_impedances": True,
                    # "smoothing_filter_size": 5,
                    # "pos_kp": 100.0,
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "mode": "absolute_pose",
                    "command_input_limits": None,
                    "command_output_limits": None,
                    # "use_impedances": True,
                    # "smoothing_filter_size": 5,
                    # "pos_kp": 100.0,
                },
                "gripper_left": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": "default",
                },
                "gripper_right": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": "default",
                },
                # "trunk": {
                #     "name": "InverseKinematicsController",
                #     "mode": "absolute_pose",
                #     "command_input_limits": None,
                #     "command_output_limits": None,
                #     "use_impedances": True,
                #     "smoothing_filter_size": 5,
                #     "pos_kp": 1.0,
                # },
                # "trunk": {
                #     "name": "InverseKinematicsController",
                # },
                "base": {
                    "name": "HolonomicBaseJointController",
                    "motor_type": motor_type,
                    "command_input_limits": None,
                    "use_impedances": False,
                },
            },
            "action_normalize": False,
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": CAMERA_HEIGHT,
                        "image_width": CAMERA_WIDTH,
                    },
                },
            },
            "reset_joint_pos": [
                0.0000,
                0.0000,
                0.000,
                0.000,
                0.000,
                0.0000,
                1.7,
                -2.3562,
                -1.0,  # -1.1 for tilt pos
                -0.0000,
                -0.000,
                0.000,
                1.8944,
                1.8945,
                -0.9848,
                -0.9849,
                1.5612,
                1.5621,
                0.9097,
                0.9096,
                -1.5544,
                -1.5545,
                0.0500,
                0.0500,
                0.0500,
                0.0500,
            ],
            # "reset_joint_pos": [
            #     0.0000,
            #     0.0000,
            #     0.000,
            #     0.000,
            #     0.000,
            #     0.0000,
            #     -1.1345,
            #     2.5,
            #     1.2,
            #     -0.0000,
            #     -0.000,
            #     0.000,
            #     1.8944,
            #     1.8945,
            #     -0.9848,
            #     -0.9849,
            #     1.5612,
            #     1.5621,
            #     0.9097,
            #     0.9096,
            #     -1.5544,
            #     -1.5545,
            #     0.0500,
            #     0.0500,
            #     0.0500,
            #     0.0500,
            # ],
        }
        cfg = dict(scene=scene_cfg, robots=[robot_cfg], task=task_cfg)
        self.env = og.Environment(cfg)

        # Create the environment
        if RECORDING_PATH is not None:
            self.env = DataCollectionWrapper(
                env=self.env,
                output_path=RECORDING_PATH,
                only_successes=False,
                use_vr=True,
            )
            self.env.is_recording = True
            KeyboardEventHandler.initialize()
            KeyboardEventHandler.add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.ENTER,
                callback_fn=self.start_recording,
            )
            KeyboardEventHandler.add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.ESCAPE,
                callback_fn=self.stop_recording,
            )

        self.robot = self.env.robots[0]

        # Hide torso/head link for better visualization
        self.robot.links["torso_link4"].visible = False

        # Let robot settle
        for _ in range(10):
            og.sim.step()

        # Initialize empty teleop action
        self.teleop_action = TeleopAction()
        self.arm_names = ["left", "right"]

        self.eyes_link = self.robot.links["eyes"]

        self._eyes_canonical_orientation = self.eyes_link.get_position_orientation()[1]

        ##################################################################

        self.use_robot_camera = use_robot_camera
        if self.use_robot_camera:
            # Grab eye cameras
            self.left_camera = self.robot.sensors[
                [key for key in self.robot.sensors.keys() if ":Camera:0" in key][0]
            ]
            self.right_camera = self.robot.sensors[
                [key for key in self.robot.sensors.keys() if ":Camera:1" in key][0]
            ]

            self.offset_left = torch.tensor([0.0, 0.0, 0.0])
            self.offset_right = torch.tensor([0.0, 0.0, 0.0])
        else:
            # Create vision sensors as left and right cameras
            self.left_camera = create_camera("/left_eye", self.robot.scene)
            self.right_camera = create_camera("/right_eye", self.robot.scene)

            self.offset_left = torch.tensor([0.01, 0.033, 0.0])
            self.offset_right = torch.tensor([0.01, -0.033, 0.0])

        ##################################################################

        # Create a permutation matrix to remap the axes
        # This matrix will transform from VR coordinates to simulation coordinates
        self.permutation_matrix = torch.tensor(
            [
                [0, -1, 0],
                [0, 0, 1],
                [-1, 0, 0],
            ],
            dtype=torch.float32,
        )

        self.left_hand_marker = None
        self.right_hand_marker = None
        self.head_marker = None
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.left_hand_marker = PrimitiveObject(
                relative_prim_path="/left_hand_marker",
                name="left_hand",
                primitive_type="Cone",
                # radius=0.03,
                scale=0.03,
                visual_only=True,
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            self.right_hand_marker = PrimitiveObject(
                relative_prim_path="/right_hand_marker",
                name="right_hand",
                primitive_type="Cone",
                # radius=0.03,
                scale=0.03,
                visual_only=True,
                rgba=[0.0, 0.0, 1.0, 1.0],
            )
            self.head_marker = PrimitiveObject(
                relative_prim_path="/head_marker",
                name="head",
                primitive_type="Cone",
                # radius=0.03,
                scale=0.03,
                visual_only=True,
                rgba=[1.0, 0.0, 0.0, 1.0],
            )
            self.env.scene.add_object(self.left_hand_marker)
            self.env.scene.add_object(self.right_hand_marker)
            self.env.scene.add_object(self.head_marker)

        # self.left_hand_pos_offset = torch.tensor([1.0, 0.0, 0.0])
        # self.right_hand_pos_offset = torch.tensor([1.0, 0.0, 0.0])
        # self.left_hand_ori_offset = T.euler2quat(
        #     torch.tensor([-math.pi, 0.0, 0.0], dtype=torch.float32)
        # )
        # self.right_hand_ori_offset = T.euler2quat(
        #     torch.tensor([math.pi, 0.0, 0.0], dtype=torch.float32)
        # )

        # self.torso_z_offset = -0.3
        self.torso_z_offset = 0.0

        self.hand_pos_offset = torch.tensor([0.3, 0.0, 0.85 + self.torso_z_offset])
        # self.hand_pos_offset = torch.tensor([0.0, 0.0, 0.5])
        self.hand_ori_offset = {
            "left": T.euler2quat(torch.tensor([0, -math.pi, 0], dtype=torch.float32)),
            "right": T.euler2quat(torch.tensor([0, -math.pi, 0], dtype=torch.float32)),
        }

        self.base_mode = base_mode

        self.base_translation_threshold = 0.1
        self.base_rotation_threshold = 0.15  # radians, approximately 8.5 degrees

        # self.base_target_position = torch.zeros(2, dtype=torch.float32)
        self.base_step_size = 0.1  # How much to increment the target position each step

        ###################################################################################
        # Hacks for cached scene
        # dishwasher = self.env.scene.object_registry("name", "dishwasher_dngvvi_0")
        # dishwasher.visual_only = True
        # dishwasher.links["link_0"].visual_only = False

        # # dishwasher base_link j_link_0 joint friction -> 0.1
        # dishwasher.joints["j_link_0"].friction = 0.1
        # # teacup 131 move to near teacup 132
        # teacup_131 = self.env.scene.object_registry("name", "teacup_131")
        # teacup_131.set_position_orientation(
        #     torch.tensor([-11.0, 0.1253, 0.3755]),
        #     torch.tensor([-1.1451e-04, 1.1063e-04, -4.4425e-01, 8.9590e-01]),
        # )
        # for carpet in self.env.scene.object_registry("category", "carpet"):
        #     carpet.visual_only = True
        # coffee_table = self.env.scene.object_registry("name", "coffee_table_gcollb_0")
        # coffee_table.links["base_link"].mass = 200.0
        # og.sim.step()
        dishwasher = self.env.scene.object_registry("name", "dishwasher_dngvvi_0")
        dishwasher.visual_only = True
        dishwasher.links["link_0"].visual_only = False

        # dishwasher base_link j_link_0 joint friction -> 0.08
        dishwasher.joints["j_link_0"].friction = 0.1
        teacup_131 = self.env.scene.object_registry("name", "teacup_131")
        teacup_131.set_position(torch.tensor([-12.5, 2.3, 0.54]))
        coffee_table = self.env.scene.object_registry("name", "coffee_table_gcollb_0")
        coffee_table.links["base_link"].mass = 200.0

        shelf = self.env.scene.object_registry("name", "shelf_owvfik_1")
        shelf.links["base_link"].mass = 200.0
        og.sim.step()
        ###################################################################################

        print("------------Teleop initialized---------------")

    def start_recording(self):
        self.env.is_recording = True
        print("------------Recording started---------------")

    def stop_recording(self):
        self.env.save_data()
        self.env.is_recording = False
        print("------------Recording stopped---------------")

    def _pose_in_robot_frame(self, pos, orn):
        """
        Get the pose in the robot frame
        Args:
            pos (th.tensor): the position in the world frame
            orn (th.tensor): the orientation in the world frame
        Returns:
            tuple(th.tensor, th.tensor): the position and orientation in the robot frame
        """
        robot_base_pos, robot_base_orn = self.robot.get_position_orientation()
        return T.relative_pose_transform(pos, orn, robot_base_pos, robot_base_orn)

    def _pose_in_torso_frame(self, pos, orn):
        torso_link_pos, torso_link_orn = self.robot.links[
            "torso_link4"
        ].get_position_orientation()
        return T.relative_pose_transform(pos, orn, torso_link_pos, torso_link_orn)

    def step(
        self,
        head_rotation,
        head_translation,
        left_pose,
        right_pose,
        left_is_grasping,
        right_is_grasping,
    ):

        # If head rotation is identity matrix, we assume the headset is not being tracked; thus we will have arm targets equal to their current positions
        if torch.all(head_rotation == torch.eye(3)):
            # left_pose = (
            #     torch.tensor([0.5, 0.2, 1.3], dtype=torch.float32),
            #     T.euler2quat(torch.tensor([math.pi / 2, 0, 0], dtype=torch.float32)),
            # )
            # right_pose = (
            #     torch.tensor([0.5, -0.2, 1.3], dtype=torch.float32),
            #     T.euler2quat(torch.tensor([-math.pi / 2, 0, 0], dtype=torch.float32)),
            # )
            # # if self.debug_mode:
            # #     self.left_hand_marker.set_position_orientation(
            # #         position=left_pose[0], orientation=left_pose[1]
            # #     )
            # #     self.right_hand_marker.set_position_orientation(
            # #         position=right_pose[0], orientation=right_pose[1]
            # #     )
            # self.teleop_action.left = torch.cat(
            #     (
            #         left_pose[0],
            #         T.quat2axisangle(left_pose[1]),
            #         torch.tensor([0], dtype=torch.float32),
            #     )
            # )
            # self.teleop_action.right = torch.cat(
            #     (
            #         right_pose[0],
            #         T.quat2axisangle(right_pose[1]),
            #         torch.tensor([0], dtype=torch.float32),
            #     )
            # )
            # self.teleop_action.torso = torch.zeros(6, dtype=torch.float32)
            # self.teleop_action.torso[2] = 1.125739
            # self.teleop_action.torso[4] = T.quat2axisangle(
            #     T.euler2quat(torch.tensor([0, -math.pi / 3, 0], dtype=torch.float32))
            # )[1]
            og.sim.step()

            left_image = self.left_camera.get_obs()[0]["rgb"].numpy()[:, :, :3]
            right_image = self.right_camera.get_obs()[0]["rgb"].numpy()[:, :, :3]
            # self.env.step(self.robot.teleop_data_to_action(self.teleop_action))

            return left_image, right_image

        # print("Head rotation: ", T.mat2euler(head_rotation).tolist())
        # print("Head translation: ", head_translation.tolist())

        # left_pose = (
        #     self.permutation_matrix @ left_pose[0],
        #     T.quat_multiply(left_pose[1], T.mat2quat(self.permutation_matrix)),
        # )
        # right_pose = (
        #     self.permutation_matrix @ right_pose[0],
        #     T.quat_multiply(right_pose[1], T.mat2quat(self.permutation_matrix)),
        # )

        if not self.use_robot_camera:
            permuted_head_rotation = (
                self.permutation_matrix @ head_rotation @ self.permutation_matrix.T
            )
            permuted_head_rotation = permuted_head_rotation @ T.euler2mat(
                torch.tensor([0.0, 0.0, 40], dtype=torch.float32)
            )
            eyes_pose = self.eyes_link.get_position_orientation()
            # curr_left_offset = self.offset_left @ head_rotation.T
            # curr_right_offset = self.offset_right @ head_rotation.T
            curr_left_offset = self.offset_left @ permuted_head_rotation.T
            curr_right_offset = self.offset_right @ permuted_head_rotation.T

            # self.left_camera.set_position_orientation(
            #     position=eyes_pose[0] + curr_left_offset,
            #     orientation=T.quat_multiply(
            #         eyes_pose[1],
            #         T.mat2quat(head_rotation),
            #     ),
            # )
            # self.right_camera.set_position_orientation(
            #     position=eyes_pose[0] + curr_right_offset,
            #     orientation=T.quat_multiply(
            #         eyes_pose[1],
            #         T.mat2quat(head_rotation),
            #     ),
            # )
            self.left_camera.set_position_orientation(
                position=eyes_pose[0] + curr_left_offset,
                orientation=T.quat_multiply(
                    eyes_pose[1],
                    T.mat2quat(permuted_head_rotation),
                ),
            )

            self.right_camera.set_position_orientation(
                position=eyes_pose[0] + curr_right_offset,
                orientation=T.quat_multiply(
                    eyes_pose[1],
                    T.mat2quat(permuted_head_rotation),
                ),
            )

        # Process hand pose data
        # for arm_name in self.arm_names:
        #     if arm_name == "left":
        #         raw_hand_pose = (
        #             left_pose[0] + self.left_hand_pos_offset,
        #             T.quat_multiply(
        #                 left_pose[1],
        #                 self.left_hand_ori_offset,
        #             ),
        #         )
        #         if self.debug_mode:
        #             self.left_hand_marker.set_position_orientation(
        #                 position=raw_hand_pose[0],
        #                 orientation=raw_hand_pose[1],
        #             )
        #     else:
        #         raw_hand_pose = (
        #             right_pose[0] + self.right_hand_pos_offset,
        #             T.quat_multiply(
        #                 right_pose[1],
        #                 self.right_hand_ori_offset,
        #             ),
        #         )
        #         if self.debug_mode:
        #             self.right_hand_marker.set_position_orientation(
        #                 position=raw_hand_pose[0],
        #                 orientation=raw_hand_pose[1],
        #             )

        #     controller_pose_in_robot_frame = self._pose_in_robot_frame(*raw_hand_pose)

        #     self.teleop_action[arm_name] = torch.cat(
        #         (
        #             controller_pose_in_robot_frame[0],
        #             T.quat2axisangle(
        #                 T.quat_multiply(
        #                     controller_pose_in_robot_frame[1],
        #                     self.robot.teleop_rotation_offset[arm_name],
        #                 )
        #             ),
        #             # TODO: fill this in
        #             # Our multi-finger gripper controller closes the gripper when the value is -1.0 and opens when > 0.0
        #             torch.tensor([0], dtype=torch.float32),
        #         )
        #     )

        # Process base pose data
        # TODO: fill these in
        # Idea: headset x-y-yaw -> base x-y-yaw; headset pitch-z -> torso pitch-z
        # For that to work, we will need to update eyes link local orientation to match head orientation

        ################################### ARM ACTION ###################################

        # robot_ori = self.robot.get_position_orientation()[1]
        left_pose = (
            left_pose[0] + self.hand_pos_offset,
            # left_pose[1],
            T.quat_multiply(left_pose[1], self.hand_ori_offset["left"]),
        )
        right_pose = (
            right_pose[0] + self.hand_pos_offset,
            # right_pose[1],
            T.quat_multiply(right_pose[1], self.hand_ori_offset["right"]),
        )

        arm_action = {"left": left_pose, "right": right_pose}

        # left_pose_in_robot_frame = self._pose_in_robot_frame(*left_pose)
        # # print("Left hand position in robot frame: ", left_pose_in_robot_frame[0])
        # right_pose_in_robot_frame = self._pose_in_robot_frame(*right_pose)
        # left_hand_in_robot_frame = (
        #     left_pose_in_robot_frame[0],  #  + self.hand_pos_offset,
        #     # T.quat_multiply(left_pose_in_robot_frame[1], self.hand_ori_offset["left"]),
        #     left_pose_in_robot_frame[1],
        # )
        # right_hand_in_robot_frame = (
        #     right_pose_in_robot_frame[0],  #  + self.hand_pos_offset,
        #     # T.quat_multiply(
        #     #     right_pose_in_robot_frame[1], self.hand_ori_offset["right"]
        #     # ),
        #     right_pose_in_robot_frame[1],
        # )
        # arm_action = {
        #     "left": left_hand_in_robot_frame,
        #     "right": right_hand_in_robot_frame,
        # }

        # # First apply offsets to raw poses
        # left_hand_world = (left_pose[0] + self.hand_pos_offset, left_pose[1])
        # right_hand_world = (right_pose[0] + self.hand_pos_offset, right_pose[1])

        # # Then transform the complete poses to robot frame
        # left_hand_in_robot_frame = self._pose_in_robot_frame(*left_hand_world)
        # right_hand_in_robot_frame = self._pose_in_robot_frame(*right_hand_world)

        # arm_action = {
        #     "left": left_hand_in_robot_frame,
        #     "right": right_hand_in_robot_frame,
        # }

        for arm_name in self.arm_names:
            # Our multi-finger gripper controller closes the gripper when the value is -1.0 and opens when > 0.0
            gripper_action = -1.0 if locals()[f"{arm_name}_is_grasping"] else 1.0
            self.teleop_action[arm_name] = torch.cat(
                (
                    arm_action[arm_name][0],
                    T.quat2axisangle(arm_action[arm_name][1]),
                    torch.tensor([gripper_action], dtype=torch.float32),
                )
            )

        ################################### BASE ACTION ###################################

        head_in_robot_frame = self._pose_in_robot_frame(
            head_translation,
            T.mat2quat(head_rotation),
        )

        if self.base_mode == "velocity":
            # DELTA VELOCITY MODE
            x_displacement = head_translation[0]
            y_displacement = head_translation[1]
            head_yaw = T.mat2euler(head_rotation)[2]
            x_vel, y_vel, yaw_vel = 0.0, 0.0, 0.0
            if abs(x_displacement) > self.base_translation_threshold:
                x_vel = x_displacement * 5.0
            if abs(y_displacement) > self.base_translation_threshold:
                y_vel = y_displacement * 5.0
            if abs(head_yaw) > self.base_rotation_threshold:
                yaw_vel = head_yaw * 2.0
            self.teleop_action.base = torch.tensor(
                [x_vel, y_vel, yaw_vel], dtype=torch.float32
            )
        elif self.base_mode == "position":
            # ABSOLUTE POSITION MODE
            self.teleop_action.base = torch.zeros(3)
            self.teleop_action.base[0] = head_in_robot_frame[0][0] * 0.5
            self.teleop_action.base[1] = head_in_robot_frame[0][1] * 0.5
            yaw = T.quat2euler(head_in_robot_frame[1])[2]
            # print("Yaw: ", yaw)
            self.teleop_action.base[2] = yaw
        elif self.base_mode == "hybrid":
            # HYBRID MODE: velocity-like control for x,y with position control for yaw
            self.teleop_action.base = torch.zeros(3)

            # Get head yaw in world frame
            head_yaw = T.mat2euler(head_rotation)[2]

            # Get displacements in world frame
            x_displacement = head_translation[0]
            y_displacement = head_translation[1]

            # Create rotation matrix for head yaw
            cos_yaw = torch.cos(
                -head_yaw
            )  # Negative because we want to rotate back to head frame
            sin_yaw = torch.sin(-head_yaw)

            # Rotate the displacement into head frame
            x_displacement_head = x_displacement * cos_yaw - y_displacement * sin_yaw
            y_displacement_head = x_displacement * sin_yaw + y_displacement * cos_yaw

            # Apply movement in robot's local frame based on head-relative displacement
            if abs(x_displacement_head) > self.base_translation_threshold:
                # print("X displacement: ", x_displacement_head)
                self.teleop_action.base[0] = self.base_step_size * torch.sign(
                    x_displacement_head
                )
            if abs(y_displacement_head) > self.base_translation_threshold:
                # print("Y displacement: ", y_displacement_head)
                self.teleop_action.base[1] = self.base_step_size * torch.sign(
                    y_displacement_head
                )

            # Use absolute position control for yaw (same as position mode)
            yaw = T.quat2euler(head_in_robot_frame[1])[2]
            self.teleop_action.base[2] = yaw
        else:
            raise ValueError("Invalid base mode")

        ################################### TORSO ACTION ###################################
        self.teleop_action.torso = torch.zeros(6, dtype=torch.float32)
        # self.teleop_action.torso[2] = 1.1 + self.torso_z_offset
        # self.teleop_action.torso[4] = T.quat2axisangle(head_in_robot_frame[1])[1]

        ################################### DEBUG ###################################

        if self.debug_mode:
            # self.head_marker.set_position_orientation(
            #     head_translation, T.mat2quat(head_rotation)
            # )
            # self.left_hand_marker.set_position_orientation(*left_pose)
            # self.right_hand_marker.set_position_orientation(*right_pose)
            robot_base_pose = self.robot.get_position_orientation()
            # print("Robot base pose: ", robot_base_pose)
            self.head_marker.set_position_orientation(
                *T.pose_transform(*robot_base_pose, *head_in_robot_frame)
            )
            # self.head_marker.set_position_orientation(*head_in_robot_frame)
            # The hand debug markers need to be transformed back into the world frame (the actions are in the robot frame)
            self.left_hand_marker.set_position_orientation(
                *T.pose_transform(*robot_base_pose, *arm_action["left"])
            )
            self.right_hand_marker.set_position_orientation(
                *T.pose_transform(*robot_base_pose, *arm_action["right"])
            )

        ################################### OBSERVATION ###################################

        # # Grab left and right images
        left_image = self.left_camera.get_obs()[0]["rgb"].numpy()[:, :, :3]
        right_image = self.right_camera.get_obs()[0]["rgb"].numpy()[:, :, :3]

        self.env.step(self.robot.teleop_data_to_action(self.teleop_action))

        return left_image, right_image

    def end(self):
        og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Teleoperation Recording")
    parser.add_argument(
        "--user_name", type=str, required=True, help="Name of the user for recording"
    )
    args = parser.parse_args()

    # Set up recording path
    BASE_RECORDING_PATH = "/home/yhang/brs_data" # TODO: Change this to your desired base path
    RECORDING_PATH = setup_recording_path(BASE_RECORDING_PATH, args.user_name)

    teleoperator = OGTeleop()
    og_sim = Sim(debug_mode=False, use_robot_camera=True, base_mode="hybrid")
    counter = 0

    try:
        while True:
            (
                head_rotation,
                head_translation,
                left_pose,
                right_pose,
                left_is_grasping,
                right_is_grasping,
            ) = teleoperator.step()

            counter += 1

            left_img, right_img = og_sim.step(
                head_rotation,
                head_translation,
                left_pose,
                right_pose,
                left_is_grasping,
                right_is_grasping,
            )
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

    except KeyboardInterrupt:
        og_sim.end()
        exit(0)
