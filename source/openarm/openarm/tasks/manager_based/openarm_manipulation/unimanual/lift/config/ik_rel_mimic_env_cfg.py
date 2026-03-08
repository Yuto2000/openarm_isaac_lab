# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IK-relative + Mimic configuration for OpenArm Lift Cube task."""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from .joint_pos_env_cfg import OpenArmCubeLiftEnvCfg


@configclass
class OpenArmCubeLiftIKRelMimicEnvCfg(OpenArmCubeLiftEnvCfg, MimicEnvCfg):
    """
    OpenArm Lift Cube environment config for Isaac Lab Mimic.

    Inherits from both OpenArmCubeLiftEnvCfg and MimicEnvCfg.
    Overrides arm_action to use IK-relative control (delta EEF pose),
    which is required for Mimic's pose transformation workflow.
    """

    def __post_init__(self):
        # Initialize parent configs
        super().__post_init__()

        # Reduce scene envs for Mimic (consolidated_demo controls num_envs via CLI)
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5

        # Add success termination (required by consolidated_demo.py)
        # Success: object is within 5cm of the goal position
        self.terminations.success = DoneTerm(
            func=mdp.object_reached_goal,
            params={"command_name": "object_pose", "threshold": 0.05},
        )

        # Switch arm action to IK-relative (delta EEF pose)
        # Action space: [delta_pos(3), delta_rot_axis_angle(3), gripper(1)] = 7D
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_joint.*"],
            body_name="openarm_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
        )

        # Data generation config
        self.datagen_config.name = "demo_src_lift_openarm"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Subtask definitions for Lift Cube:
        #   Subtask 1: Reach → Grasp cube (detected by grasped_cube signal)
        #   Subtask 2: Lift cube to target (final, no term signal)
        subtask_configs = [
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="grasped_cube",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Reach and grasp cube",
                next_subtask_description="Lift cube to target position",
            ),
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,  # final subtask
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift cube to target position",
            ),
        ]
        # Key must match eef_name used in lift_mimic_env.py
        self.subtask_configs["end_effector"] = subtask_configs


@configclass
class OpenArmCubeLiftIKRelMimicEnvCfg_PLAY(OpenArmCubeLiftIKRelMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
