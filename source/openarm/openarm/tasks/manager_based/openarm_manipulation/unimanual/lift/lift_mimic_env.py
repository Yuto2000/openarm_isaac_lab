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

"""Isaac Lab Mimic environment for OpenArm Lift Cube task."""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class OpenArmLiftMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper for OpenArm Lift Cube task.

    Uses IK-relative actions (delta EEF pose) compatible with Isaac Lab Mimic
    data generation workflow.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current end-effector pose from the FrameTransformer sensor.

        Args:
            eef_name: Name of the end effector (unused, single EEF).
            env_ids: Environment indices. If None, all envs.

        Returns:
            EEF pose matrices of shape (len(env_ids), 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        ee_frame = self.scene["ee_frame"]
        # target_pos_w: (num_envs, num_targets, 3)
        eef_pos = ee_frame.data.target_pos_w[env_ids, 0, :]
        # target_quat_w: (num_envs, num_targets, 4) in wxyz format
        eef_quat = ee_frame.data.target_quat_w[env_ids, 0, :]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Convert target EEF pose to IK-relative delta action.

        For IK-relative control:
          action = [delta_pos(3), delta_rot_axis_angle(3), gripper(1)]

        Args:
            target_eef_pose_dict: Dict of 4x4 target EEF pose.
            gripper_action_dict: Dict of gripper actions.
            action_noise_dict: Optional noise scales per EEF.
            env_id: Environment index.

        Returns:
            Action tensor of shape (7,).
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # Delta position
        delta_position = target_pos - curr_pos

        # Delta rotation as axis-angle
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)

        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convert IK-relative action back to target EEF pose.

        Args:
            action: Actions of shape (num_envs, action_dim).

        Returns:
            Dict of target EEF pose tensors (num_envs, 4, 4).
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        # Convert axis-angle delta to rotation matrix
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / (delta_rotation_angle + 1e-8)
        is_close_to_zero = torch.isclose(
            delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
        ).squeeze(1)
        delta_rotation_axis[is_close_to_zero] = 0.0

        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis
        )
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract gripper action from action tensor (last dimension).

        Args:
            actions: Actions of shape (num_envs, num_steps, action_dim).

        Returns:
            Dict of gripper action tensors.
        """
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Detect subtask completion signals.

        For Lift Cube:
          - "grasped_cube": gripper mostly closed AND cube slightly above table

        Args:
            env_ids: Environment indices. If None, all envs.

        Returns:
            Dict of termination signals (float 0.0 or 1.0).
        """
        if env_ids is None:
            env_ids = slice(None)

        robot = self.scene["robot"]
        obj = self.scene["object"]

        # Gripper finger joint positions (open=0.044, closed≈0.0)
        finger_joint_ids, _ = robot.find_joints("openarm_finger_joint.*")
        finger_pos = robot.data.joint_pos[env_ids][:, finger_joint_ids]
        # Grasping: fingers mostly closed
        gripper_closed = finger_pos.mean(dim=-1) < 0.015

        # Object height: table surface ≈ z=0 in robot root frame
        obj_pos_z = obj.data.root_pos_w[env_ids, 2]
        # Consider grasped when cube is lifted at least 2cm above rest position
        obj_lifted = obj_pos_z > 0.07

        grasped = (gripper_closed & obj_lifted).float()

        return {"grasped_cube": grasped}
