import torch

from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot


class D1Robot(LeggedRobot):
    """D1-specific locomotion hooks.

    Keep the shared LeggedRobot implementation close to upstream HIMLoco and put
    D1 reward/control experiments here so other robots are not affected.
    """

    def _command_moving_mask(self, lin_thresh=0.1, yaw_thresh=0.1):
        lin_cmd = torch.norm(self.commands[:, :2], dim=1)
        yaw_cmd = torch.abs(self.commands[:, 2])
        return torch.logical_or(lin_cmd > lin_thresh, yaw_cmd > yaw_thresh)

    def _get_gait_phase(self):
        return (self.episode_length_buf.float() * self.dt * self.cfg.rewards.gait_frequency) % 1.0

    def _get_phase_masks(self):
        phase = self._get_gait_phase()
        offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        leg_phase = (phase[:, None] + offsets[None, :]) % 1.0

        duty = self.cfg.rewards.duty_factor
        margin = self.cfg.rewards.phase_transition_margin
        desired_contact = leg_phase < duty
        desired_swing = ~desired_contact

        stance_core = (leg_phase > margin) & (leg_phase < duty - margin)
        swing_core = (leg_phase > duty + margin) & (leg_phase < 1.0 - margin)
        valid = stance_core | swing_core
        return desired_contact, desired_swing, stance_core, swing_core, valid

    def _get_feet_contact_bool(self):
        return self.contact_forces[:, self.feet_indices, 2] > 1.0

    def _forward_straight_mask(self):
        return (
            (self.commands[:, 0] > self.cfg.rewards.forward_cmd_min)
            & (torch.abs(self.commands[:, 1]) < self.cfg.rewards.forward_lateral_cmd_max)
            & (torch.abs(self.commands[:, 2]) < self.cfg.rewards.forward_yaw_cmd_max)
        )

    def _get_forward_contact_bool(self):
        threshold = self.cfg.rewards.contact_force_threshold
        return self.contact_forces[:, self.feet_indices, 2] > threshold

    def _update_forward_swing_peaks(self):
        if getattr(self, "_forward_swing_peak_update_step", None) == self.common_step_counter:
            return
        self._forward_swing_peak_update_step = self.common_step_counter

        contact = self._get_forward_contact_bool()
        feet_height = self._get_feet_heights()
        forward_mask = self._forward_straight_mask()

        if not hasattr(self, "forward_swing_peak_current"):
            self.forward_swing_peak_current = feet_height.clone()
            self.forward_swing_peak_completed = feet_height.clone()
            self.forward_swing_peak_valid = torch.zeros_like(contact, dtype=torch.bool)
            self.forward_swing_peak_last_contacts = contact.clone()
            return

        forward = forward_mask.unsqueeze(1)
        swing = (~contact) & forward
        touchdown = contact & (~self.forward_swing_peak_last_contacts) & forward

        self.forward_swing_peak_current = torch.where(
            swing,
            torch.maximum(self.forward_swing_peak_current, feet_height),
            self.forward_swing_peak_current,
        )
        self.forward_swing_peak_completed = torch.where(
            touchdown,
            self.forward_swing_peak_current,
            self.forward_swing_peak_completed,
        )
        self.forward_swing_peak_valid = torch.where(
            touchdown,
            torch.ones_like(self.forward_swing_peak_valid),
            self.forward_swing_peak_valid,
        )
        self.forward_swing_peak_current = torch.where(
            touchdown,
            feet_height,
            self.forward_swing_peak_current,
        )

        reset = (~forward_mask).unsqueeze(1)
        self.forward_swing_peak_current = torch.where(reset, feet_height, self.forward_swing_peak_current)
        self.forward_swing_peak_completed = torch.where(reset, feet_height, self.forward_swing_peak_completed)
        self.forward_swing_peak_valid = torch.where(
            reset,
            torch.zeros_like(self.forward_swing_peak_valid),
            self.forward_swing_peak_valid,
        )
        self.forward_swing_peak_last_contacts = contact

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)

        if not hasattr(self, "zero_command_mask"):
            self.zero_command_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.zero_command_mask[env_ids] = False

        backward_prob = getattr(self.cfg.commands, "backward_command_prob", 0.0)
        if backward_prob > 0.0 and env_ids.numel() > 0:
            backward_mask = torch.rand(env_ids.numel(), device=self.device) < backward_prob
            backward_env_ids = env_ids[backward_mask]
            if backward_env_ids.numel() > 0:
                backward_min = self.command_ranges["lin_vel_x"][0]
                backward_max = min(-0.3, self.command_ranges["lin_vel_x"][1])
                if backward_min < backward_max:
                    self.commands[backward_env_ids, 0] = torch_rand_float(
                        backward_min,
                        backward_max,
                        (backward_env_ids.numel(), 1),
                        device=self.device,
                    ).squeeze(1)
                    self.commands[backward_env_ids, 1:3] = 0.0
                    if self.cfg.commands.heading_command:
                        forward = quat_apply(self.base_quat[backward_env_ids], self.forward_vec[backward_env_ids])
                        self.commands[backward_env_ids, 3] = torch.atan2(forward[:, 1], forward[:, 0])

        zero_prob = getattr(self.cfg.commands, "zero_command_prob", 0.0)
        if zero_prob <= 0.0 or env_ids.numel() == 0:
            return

        zero_mask = torch.rand(env_ids.numel(), device=self.device) < zero_prob
        zero_env_ids = env_ids[zero_mask]
        if zero_env_ids.numel() == 0:
            return

        self.commands[zero_env_ids, :3] = 0.0
        self.zero_command_mask[zero_env_ids] = True
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[zero_env_ids], self.forward_vec[zero_env_ids])
            self.commands[zero_env_ids, 3] = torch.atan2(forward[:, 1], forward[:, 0])

    def _process_dof_props(self, props, env_id):
        dof_armature = getattr(self.cfg.asset, "dof_armature", None)
        if dof_armature is not None:
            for i in range(len(props)):
                props["armature"][i] = dof_armature

        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.original_armature = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.original_armature[i] = props["armature"][i].item()

        if getattr(self.cfg.domain_rand, "randomize_joint_armature", False):
            if not hasattr(self, "joint_armature_coeffs"):
                self.joint_armature_coeffs = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.joint_armature_coeffs[env_id, :] = torch_rand_float(
                self.cfg.domain_rand.joint_armature_range[0],
                self.cfg.domain_rand.joint_armature_range[1],
                (1, self.num_dof),
                device=self.device,
            )
            for i in range(len(props)):
                props["armature"][i] = (self.original_armature[i] * self.joint_armature_coeffs[env_id, i]).item()

        return props

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_reduction
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        effective_limits = self.torque_limits * self.motor_strength_factors
        if not hasattr(self, "torque_saturation_count"):
            self.torque_saturation_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_saturation_count += (torch.abs(torques) >= effective_limits * 0.98).any(dim=1).float()
        return torch.clip(torques, -effective_limits, effective_limits)

    def reset_idx(self, env_ids):
        has_sat_count = hasattr(self, "torque_saturation_count")
        if has_sat_count and env_ids.numel() > 0:
            ep_len = self.episode_length_buf[env_ids].float().clamp(min=1)
            torque_saturation_rate = torch.mean(self.torque_saturation_count[env_ids] / ep_len)

        super().reset_idx(env_ids)

        if has_sat_count and env_ids.numel() > 0:
            self.extras["episode"]["torque_saturation_rate"] = torque_saturation_rate
            self.torque_saturation_count[env_ids] = 0.0
        if hasattr(self, "swing_peak_height"):
            self.swing_peak_height[env_ids, :] = 0.0
        if hasattr(self, "swing_peak_last_contacts"):
            self.swing_peak_last_contacts[env_ids, :] = False
        if hasattr(self, "touchdown_last_contacts"):
            self.touchdown_last_contacts[env_ids, :] = False
        if hasattr(self, "forward_diagonal_peak_height"):
            self.forward_diagonal_peak_height[env_ids, :] = 0.0
        if hasattr(self, "forward_diagonal_completed_peak"):
            self.forward_diagonal_completed_peak[env_ids, :] = 0.0
        if hasattr(self, "forward_diagonal_last_contacts"):
            self.forward_diagonal_last_contacts[env_ids, :] = False
        if hasattr(self, "forward_swing_peak_current"):
            feet_height = self._get_feet_heights()
            self.forward_swing_peak_current[env_ids, :] = feet_height[env_ids, :]
        if hasattr(self, "forward_swing_peak_completed"):
            feet_height = self._get_feet_heights()
            self.forward_swing_peak_completed[env_ids, :] = feet_height[env_ids, :]
        if hasattr(self, "forward_swing_peak_valid"):
            self.forward_swing_peak_valid[env_ids, :] = False
        if hasattr(self, "forward_swing_peak_last_contacts"):
            self.forward_swing_peak_last_contacts[env_ids, :] = self._get_forward_contact_bool()[env_ids, :]

    def _reward_lin_vel_z(self):
        return torch.sqrt(torch.square(self.base_lin_vel[:, 2]) + 1e-6)

    def _reward_ang_vel_xy(self):
        return torch.sqrt(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) + 1e-6)

    def _reward_action_rate(self):
        return torch.sqrt(torch.sum(torch.square(self.last_actions - self.actions), dim=1) + 1e-6)

    def _reward_smoothness(self):
        return torch.sqrt(torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1) + 1e-6)

    def _reward_dof_vel(self):
        return torch.sqrt(torch.sum(torch.square(self.dof_vel), dim=1) + 1e-6)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt

        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - 0.15) * first_contact, dim=1)
        rew_air_time *= self._command_moving_mask(lin_thresh=0.1, yaw_thresh=0.1).float()
        self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_stand_still(self):
        stand_mask = ~self._command_moving_mask(lin_thresh=0.1, yaw_thresh=0.1)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * stand_mask.float()

    def _reward_zero_hip_target_dev(self):
        hip_ids = [0, 3, 6, 9]
        if hasattr(self, "zero_command_mask"):
            zero_cmd = self.zero_command_mask
        else:
            zero_cmd = ~self._command_moving_mask(lin_thresh=0.05, yaw_thresh=0.05)

        if hasattr(self, "joint_pos_target"):
            hip_target_dev = self.joint_pos_target[:, hip_ids] - self.default_dof_pos[:, hip_ids]
        else:
            hip_action_scale = self.cfg.control.action_scale * self.cfg.control.hip_reduction
            hip_target_dev = hip_action_scale * self.actions[:, hip_ids]
        return torch.sum(torch.square(hip_target_dev), dim=1) * zero_cmd.float()

    def _reward_backward_base_height(self):
        backward = self.commands[:, 0] < -0.3
        base_height = self._get_base_heights()
        target = self.cfg.rewards.backward_base_height_target
        low_error = torch.clamp(target - base_height, min=0.0)
        return torch.square(low_error) * backward.float()

    def _reward_backward_base_height_floor(self):
        backward = self.commands[:, 0] < -0.3
        base_height = self._get_base_heights()
        target = self.cfg.rewards.backward_base_height_floor_target
        margin = max(self.cfg.rewards.backward_base_height_floor_margin, 1e-6)
        # Backward gait learned a low crouch; normalize the centimeter-level height error so it is visible.
        low_error = torch.clamp(target - base_height, min=0.0)
        normalized_error = low_error / margin
        return torch.square(normalized_error) * backward.float()

    def _reward_moving_base_height_floor(self):
        moving = self._command_moving_mask(lin_thresh=0.15, yaw_thresh=0.15)
        base_height = self._get_base_heights()
        target = self.cfg.rewards.moving_base_height_floor_target
        margin = max(self.cfg.rewards.moving_base_height_floor_margin, 1e-6)
        # Normalize the low-height error so centimeter-level gait drops have a usable reward scale.
        low_error = torch.clamp(target - base_height, min=0.0)
        normalized_error = low_error / margin
        return torch.square(normalized_error) * moving.float()

    def _reward_base_height_band(self):
        base_height = self._get_base_heights()
        min_height = self.cfg.rewards.base_height_band_min
        max_height = self.cfg.rewards.base_height_band_max
        margin = max(self.cfg.rewards.base_height_band_margin, 1e-6)

        # D1 accepts a 0.36-0.38 m body-height band; low crouches are more harmful than slight overshoot.
        low_error = torch.clamp(min_height - base_height, min=0.0)
        high_error = torch.clamp(base_height - max_height, min=0.0)
        normalized_error = (low_error + 0.5 * high_error) / margin
        return torch.square(normalized_error)

    def _reward_backward_base_height_band(self):
        backward = self.commands[:, 0] < -0.3
        base_height = self._get_base_heights()
        min_height = self.cfg.rewards.backward_base_height_band_min
        margin = max(self.cfg.rewards.backward_base_height_band_margin, 1e-6)

        # Backward gait is the remaining low-crouch case, so only add extra pressure below the acceptable floor.
        low_error = torch.clamp(min_height - base_height, min=0.0)
        normalized_error = low_error / margin
        return torch.square(normalized_error) * backward.float()

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_speed_norm = torch.norm(self.feet_vel[:, :, :2], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_joint_pos_penalty(self):
        moving_scale = 0.7
        velocity_threshold = 0.5
        yaw_vel_threshold = 0.5

        body_vel = torch.linalg.norm(self.base_lin_vel[:, :2], dim=1)
        body_yaw_vel = torch.abs(self.base_ang_vel[:, 2])
        cmd_moving = self._command_moving_mask(lin_thresh=0.1, yaw_thresh=0.1)

        error = torch.linalg.norm(
            self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]],
            dim=1,
        )
        moving = torch.logical_or(cmd_moving, torch.logical_or(body_vel > velocity_threshold, body_yaw_vel > yaw_vel_threshold))
        return torch.where(moving, moving_scale * error, error)

    def _reward_swing_clearance(self):
        moving = self._command_moving_mask(lin_thresh=0.1, yaw_thresh=0.1)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        swing = ~contact
        feet_height = self._get_feet_heights()

        target = self.cfg.rewards.swing_clearance_target
        err = torch.clamp(target - feet_height, min=0.0)
        penalty = torch.sum(torch.square(err) * swing.float(), dim=1)
        return penalty * moving.float()

    def _reward_swing_peak_height_band(self):
        moving = self._command_moving_mask(lin_thresh=0.1, yaw_thresh=0.1)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        if not hasattr(self, "swing_peak_height"):
            self.swing_peak_height = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
            self.swing_peak_last_contacts = contact.clone()
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        first_contact = contact & (~self.swing_peak_last_contacts)
        swing = ~contact
        feet_height = self._get_feet_heights()
        self.swing_peak_height = torch.where(swing, torch.maximum(self.swing_peak_height, feet_height), self.swing_peak_height)

        min_height = self.cfg.rewards.swing_peak_height_min
        max_height = self.cfg.rewards.swing_peak_height_max
        margin = max(self.cfg.rewards.swing_peak_height_margin, 1e-6)
        low_error = torch.clamp(min_height - self.swing_peak_height, min=0.0)
        high_error = torch.clamp(self.swing_peak_height - max_height, min=0.0)
        normalized_error = (low_error + high_error) / margin
        penalty = torch.sum(torch.square(normalized_error) * first_contact.float(), dim=1)

        self.swing_peak_height = torch.where(contact, torch.zeros_like(self.swing_peak_height), self.swing_peak_height)
        self.swing_peak_last_contacts = contact
        return penalty * moving.float()

    def _reward_touchdown_impact(self):
        moving = self._command_moving_mask(lin_thresh=0.1, yaw_thresh=0.1)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        if not hasattr(self, "touchdown_last_contacts"):
            self.touchdown_last_contacts = contact.clone()
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        first_contact = contact & (~self.touchdown_last_contacts)
        downward_speed = torch.clamp(-self.feet_vel[:, :, 2] - self.cfg.rewards.touchdown_vel_threshold, min=0.0)
        penalty = torch.sum(torch.square(downward_speed) * first_contact.float(), dim=1)

        self.touchdown_last_contacts = contact
        return penalty * moving.float()

    def _reward_forward_diagonal_swing_peak_balance(self):
        straight_forward = (
            (self.commands[:, 0] > 0.3)
            & (torch.abs(self.commands[:, 1]) < 0.1)
            & (torch.abs(self.commands[:, 2]) < 0.2)
        )
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        if not hasattr(self, "forward_diagonal_peak_height"):
            self.forward_diagonal_peak_height = torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.forward_diagonal_completed_peak = torch.zeros_like(self.forward_diagonal_peak_height)
            self.forward_diagonal_last_contacts = contact.clone()
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        first_contact = contact & (~self.forward_diagonal_last_contacts)
        swing = ~contact
        feet_height = self._get_feet_heights()
        self.forward_diagonal_peak_height = torch.where(
            swing,
            torch.maximum(self.forward_diagonal_peak_height, feet_height),
            self.forward_diagonal_peak_height,
        )

        completed_peak = torch.where(first_contact, self.forward_diagonal_peak_height, self.forward_diagonal_completed_peak)
        diagonal_a = 0.5 * (completed_peak[:, 0] + completed_peak[:, 3])  # FL + RR
        diagonal_b = 0.5 * (completed_peak[:, 1] + completed_peak[:, 2])  # FR + RL

        margin = max(self.cfg.rewards.forward_diagonal_swing_peak_balance_margin, 1e-6)
        ready = torch.all(completed_peak > 0.0, dim=1)
        event = torch.any(first_contact, dim=1)
        normalized_error = torch.abs(diagonal_a - diagonal_b) / margin
        penalty = torch.square(normalized_error) * ready.float() * event.float() * straight_forward.float()

        self.forward_diagonal_completed_peak = completed_peak
        self.forward_diagonal_peak_height = torch.where(contact, torch.zeros_like(self.forward_diagonal_peak_height), self.forward_diagonal_peak_height)
        self.forward_diagonal_last_contacts = contact
        return penalty

    def _reward_forward_swing_peak_spread(self):
        self._update_forward_swing_peaks()
        if not hasattr(self, "forward_swing_peak_completed"):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        forward_mask = self._forward_straight_mask()
        valid_env = torch.all(self.forward_swing_peak_valid, dim=1)
        peaks = self.forward_swing_peak_completed
        spread = torch.max(peaks, dim=1).values - torch.min(peaks, dim=1).values

        margin = max(self.cfg.rewards.forward_swing_peak_spread_margin, 1e-6)
        err = torch.clamp(spread - margin, min=0.0) / margin
        err = torch.clamp(err, max=3.0)
        return err * err * forward_mask.float() * valid_env.float()

    def _reward_forward_diagonal_internal_swing_peak_balance(self):
        self._update_forward_swing_peaks()
        if not hasattr(self, "forward_swing_peak_completed"):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        forward_mask = self._forward_straight_mask()
        valid_env = torch.all(self.forward_swing_peak_valid, dim=1)
        peaks = self.forward_swing_peak_completed

        diff_a = torch.abs(peaks[:, 0] - peaks[:, 3])  # FL vs RR
        diff_b = torch.abs(peaks[:, 1] - peaks[:, 2])  # FR vs RL
        margin = max(self.cfg.rewards.forward_diagonal_internal_swing_peak_balance_margin, 1e-6)
        err_a = torch.clamp(diff_a - margin, min=0.0) / margin
        err_b = torch.clamp(diff_b - margin, min=0.0) / margin
        err_a = torch.clamp(err_a, max=3.0)
        err_b = torch.clamp(err_b, max=3.0)
        err = 0.5 * (err_a * err_a + err_b * err_b)
        return err * forward_mask.float() * valid_env.float()

    def _reward_forward_swing_height_cap(self):
        self._update_forward_swing_peaks()
        if not hasattr(self, "forward_swing_peak_completed"):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        forward_mask = self._forward_straight_mask()
        valid_env = torch.all(self.forward_swing_peak_valid, dim=1)
        peak_max = torch.max(self.forward_swing_peak_completed, dim=1).values

        cap = self.cfg.rewards.forward_swing_height_cap_height
        margin = max(self.cfg.rewards.forward_swing_height_cap_margin, 1e-6)
        err = torch.clamp(peak_max - cap, min=0.0) / margin
        err = torch.clamp(err, max=3.0)
        return err * err * forward_mask.float() * valid_env.float()

    def _reward_forward_min_contact_count(self):
        forward_mask = self._forward_straight_mask()
        contact_count = self._get_forward_contact_bool().float().sum(dim=1)
        min_contacts = self.cfg.rewards.min_forward_contacts
        err = torch.clamp(min_contacts - contact_count, min=0.0)
        return err * err * forward_mask.float()

    def _reward_forward_base_vertical_velocity(self):
        forward_mask = self._forward_straight_mask()
        target = max(self.cfg.rewards.forward_base_vertical_velocity_target, 1e-6)
        vertical_velocity = self.root_states[:, 9]
        err = torch.clamp(torch.abs(vertical_velocity) - target, min=0.0) / target
        err = torch.clamp(err, max=3.0)
        return err * err * forward_mask.float()

    def _reward_torque_saturation(self):
        torque_limits = torch.clamp(self.torque_limits, min=1e-6).unsqueeze(0)
        if hasattr(self, "motor_strength_factors"):
            torque_limits = torque_limits * self.motor_strength_factors

        threshold = self.cfg.rewards.torque_saturation_threshold
        denom = max(1.0 - threshold, 1e-6)
        saturation = torch.abs(self.torques) / torque_limits
        err = torch.clamp(saturation - threshold, min=0.0) / denom
        err = torch.clamp(err, max=3.0)

        mean_err = torch.mean(err * err, dim=1)
        max_err = torch.max(err, dim=1).values ** 2
        return 0.7 * mean_err + 0.3 * max_err

    def _reward_phase_contact(self):
        moving = self._command_moving_mask(lin_thresh=0.15, yaw_thresh=0.15)
        desired_contact, _, _, _, valid = self._get_phase_masks()
        contact = self._get_feet_contact_bool()
        mismatch = torch.abs(desired_contact.float() - contact.float())
        valid_count = torch.sum(valid.float(), dim=1).clamp(min=1.0)
        penalty = torch.sum(mismatch * valid.float(), dim=1) / valid_count
        return penalty * moving.float()

    def _reward_phase_swing_clearance(self):
        moving = self._command_moving_mask(lin_thresh=0.15, yaw_thresh=0.15)
        _, _, _, swing_core, _ = self._get_phase_masks()
        feet_height = self._get_feet_heights()

        target = self.cfg.rewards.phase_swing_clearance_target
        margin = max(self.cfg.rewards.phase_swing_clearance_margin, 1e-6)
        err = torch.clamp(target - feet_height, min=0.0)
        normalized_error = err / margin
        penalty = torch.sum(torch.square(normalized_error) * swing_core.float(), dim=1)
        return penalty * moving.float()
