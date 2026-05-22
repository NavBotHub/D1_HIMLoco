# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class D1RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            max_gpu_contact_pairs = 2**25
            num_position_iterations = 6
            max_depenetration_velocity = 2.0
            default_buffer_size_multiplier = 8

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.49]
        default_joint_angles = {
            # 0.37 m stance. Keep hips neutral; use mild front/rear thigh offset.
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": -0.70,
            "FL_calf_joint": -0.75,

            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.70,
            "FR_calf_joint": -0.75,

            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.80,
            "RL_calf_joint": -0.75,

            "RR_hip_joint": 0.0,
            "RR_thigh_joint": -0.80,
            "RR_calf_joint": -0.75,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 50.0}
        damping = {"joint": 2.5}
        action_scale = 0.25
        decimation = 4
        hip_reduction = 0.3

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.6
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        zero_command_prob = 0.1
        backward_command_prob = 0.0  # AB knob: set to 0.3 to oversample straight backward commands.

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/d1/urdf/d1_description.urdf"
        name = "d1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1
        flip_visual_attachments = False
        dof_armature = 0.13108
        # fix_base_link = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.2, 1.0]
        randomize_initial_joint_pos = False
        initial_joint_pos_range = [-0.05, 0.05]
        disturbance_range = [-15.0, 15.0]
        randomize_joint_armature = True
        joint_armature_range = [0.8, 1.2]

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.5
            ang_vel_xy = -0.08
            orientation = -0.2
            dof_acc = -1e-7
            joint_power = -2e-5
            # A violent band stress test: verify whether strong low-height penalties can force a 0.36 m+ moving gait.
            backward_base_height = 0.0
            backward_base_height_floor = 0.0
            backward_base_height_band = -0.05
            base_height = -5.0
            base_height_band = -0.60
            moving_base_height_floor = 0.0
            foot_clearance = -0.0
            # Keep clearance and phase off; first make the natural forward gait low-bounce and consistent.
            swing_clearance = 0.0
            swing_peak_height_band = 0.0
            touchdown_impact = -0.03
            forward_diagonal_swing_peak_balance = 0.0
            forward_swing_peak_spread = -0.04
            forward_diagonal_internal_swing_peak_balance = -0.035
            forward_swing_height_cap = -0.015
            forward_min_contact_count = -0.16
            forward_base_vertical_velocity = -0.06
            phase_contact = 0.0
            phase_swing_clearance = 0.0
            action_rate = -0.08
            smoothness = -0.02
            feet_air_time = 0.0
            collision = -1.0
            feet_stumble = -0.0
            stand_still = -0.2
            zero_hip_target_dev = -2.0  # A baseline; B can set -1.0 to test a lighter zero-hip hold.
            torques = -1e-5
            dof_vel = -0.01
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            torque_saturation = -0.045
            feet_contact_forces = 0.0
            foot_slip = -0.05
            joint_pos_penalty = -0.25

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.85
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.37
        backward_base_height_target = 0.36
        backward_base_height_floor_target = 0.355
        backward_base_height_floor_margin = 0.05
        base_height_band_min = 0.36
        base_height_band_max = 0.38
        base_height_band_margin = 0.05
        backward_base_height_band_min = 0.355
        backward_base_height_band_margin = 0.05
        moving_base_height_floor_target = 0.365
        moving_base_height_floor_margin = 0.05
        gait_frequency = 2.2
        duty_factor = 0.55
        phase_transition_margin = 0.05
        phase_swing_clearance_target = 0.055
        phase_swing_clearance_margin = 0.05
        max_contact_force = 400.0
        clearance_height_target = -0.30
        swing_clearance_target = 0.06
        swing_peak_height_min = 0.035
        swing_peak_height_max = 0.060
        swing_peak_height_margin = 0.020
        touchdown_vel_threshold = 0.30
        forward_diagonal_swing_peak_balance_margin = 0.020
        forward_cmd_min = 0.3
        forward_lateral_cmd_max = 0.1
        forward_yaw_cmd_max = 0.2
        contact_force_threshold = 5.0
        min_forward_contacts = 2
        forward_swing_peak_spread_margin = 0.010
        forward_diagonal_internal_swing_peak_balance_margin = 0.010
        forward_swing_height_cap_height = 0.060
        forward_swing_height_cap_margin = 0.020
        forward_base_vertical_velocity_target = 0.15
        torque_saturation_threshold = 0.85


class D1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_d1"
        save_interval = 1000
        max_iterations = 5000
