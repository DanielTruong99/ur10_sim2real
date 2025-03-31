# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort: skip
import ur10_sim2real.tasks.manager_based.ur10_sim2real.mdp as custom_mdp  # isort: skip

##
# Environment configuration
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action =  mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=1.0, 
        use_default_offset=True
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.5, 1.5),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "std": 0.25, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=custom_mdp.orientation_error_tanh,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"),  "std": 1.2, "command_name": "ee_pose"},
    )
    orientation_bonus = RewTerm(
        func=custom_mdp.position_orientation_bonus,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    )

    # penalties
    ee_lin_velocity = RewTerm(
        func=custom_mdp.body_lin_vel_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
    )
    ee_ang_velocity = RewTerm(
        func=custom_mdp.body_ang_vel_l2,
        weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-7e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObsGroup):
        """Observations for debug group."""

        # observation terms (order preserved)
        ee_v = ObsTerm(func=custom_mdp.body_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link")})
        ee_w = ObsTerm(
            func=custom_mdp.body_ang_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link")}
        )

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    debug: DebugCfg = DebugCfg()



@configclass
class UR10ReachEnvCfg(ReachEnvCfg):

    rewards: RewardsCfg = RewardsCfg()
    event: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set the simulation parameters
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 10.0

        # switch robot to ur10
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = None 
        self.scene.ground = None
        self.scene.light = None

        # disable the curriculum
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # enable debug visualization for ee pose command
        self.commands.ee_pose.debug_vis = True
