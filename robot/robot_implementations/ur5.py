from typing import Union
import numpy as np
import pybullet as pyb
from robot.robot import Robot
from util.rrt import bi_rrt
from time import process_time

__all__ = [
    'UR5',
    'UR5_P2P',
    'UR5_RRT'
]

class UR5(Robot):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta: float, rpy_delta: float):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, xyz_delta, rpy_delta)
        self.joints_limits_lower = np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joints_limits_upper = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.array([10., 10., 10., 10., 10., 10.])

        self.end_effector_link_id = 7
        self.base_link_id = 1

        self.urdf_path = "robots/predefined/ur5/urdf/ur5.urdf"

    def get_action_space_dims(self):
        return (6,6)  # 6 joints

    def build(self):

        self.object_id = pyb.loadURDF(self.urdf_path, basePosition=self.base_position.tolist(), baseOrientation=self.base_orientation.tolist(), useFixedBase=True)
        joints_info = [pyb.getJointInfo(self.object_id, i) for i in range(pyb.getNumJoints(self.object_id))]
        self.joints_ids = np.array([j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])

        self.moveto_joints(self.resting_pose_angles, False)     


    def _solve_ik(self, xyz: np.ndarray, quat:Union[np.ndarray, None]):
        """
        Solves the UR5's inverse kinematics for the desired pose.
        Returns the joint angles required.
        This specific implementation for the UR5 projects the frequent out of bounds
        solutions back into the allowed joint range by exploiting the
        periodicity of the 2 pi range.

        :param xyz: Vector containing the desired xyz position of the end effector.
        :param quat: Vector containing the desired rotation of the end effector.
        :return: Vector containing the joint angles required to reach the pose.
        """
        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.object_id,
            endEffectorLinkIndex=self.end_effector_link_id,
            targetPosition=xyz.tolist(),
            targetOrientation=quat.tolist() if quat is not None else None,
            lowerLimits=self.joints_limits_lower.tolist(),
            upperLimits=self.joints_limits_upper.tolist(),
            jointRanges=self.joints_range.tolist(),
            maxNumIterations=100,
            residualThreshold=.01)
        joints = np.float32(joints)
        #joints = (joints + self.joints_limits_upper) % (self.joints_range) - self.joints_limits_upper  # projects out of bounds angles back to angles within the allowed joint range
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints


class UR5_P2P(UR5):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta: float, rpy_delta: float, joints_limits_upper: list = None, joints_limits_lower: list = None ):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, xyz_delta, rpy_delta)
        self.joints_limits_lower = np.array([-np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi]) if joints_limits_lower is None else joints_limits_lower
        self.joints_limits_upper = np.array([np.pi, np.pi/2, np.pi, np.pi, np.pi, np.pi]) if joints_limits_upper is None else joints_limits_upper
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.array([10., 10., 10., 10., 10., 10.])

        self.end_effector_link_id = 7
        self.base_link_id = 1

    def _get_random_valid_angles(self):
        """
        Returns a valid set of angles for the robot.
        """
        return np.array([np.random.uniform(low, high) for low, high in zip(self.joints_limits_lower, self.joints_limits_upper)])

    def build(self):
        self.resting_pose_angles = self._get_random_valid_angles()
        super().build()
        
class UR5_RRT(UR5):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], trajectory_tolerance: float, rrt_config: dict):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, 1, 0, 0)
        self.rrt_config = rrt_config
        self.trajectory_tolerance = trajectory_tolerance

    def build(self):
        super().build()
        self.planned_trajectory = []
        self.current_sub_goal = None


    def get_action_space_dims(self):
        return (1, 1)

    def process_action(self, action: np.ndarray):
        cpu_epoch = process_time()
        # if we are calling this for the first time this episode we need to plan the trajectory
        if not self.planned_trajectory:
            q_start = self.joints_sensor.joints_angles
            goal_xyz = self.world.position_targets[self.id]
            goal_quat = self.world.rotation_targets[self.id] if self.world.rotation_targets else None
            q_goal = self._solve_ik(goal_xyz, goal_quat)

            self.planned_trajectory = bi_rrt(q_start=q_start,
                                             q_goal=q_goal,
                                             robot=self,
                                             obstacles_ids=self.world.objects_ids,
                                             max_steps=self.rrt_config["max_steps"],
                                             epsilon=self.rrt_config["epsilon"],
                                             goal_bias=self.rrt_config["goal_bias"]
                                             )
            
            if not self.planned_trajectory:
                raise Exception("RRT planning did not find solution after " + str(self.rrt_config["max_steps"]) + " steps!")
            self.current_sub_goal = self.planned_trajectory.pop(0)
            # reset robot after planning
            self.moveto_joints(q_start, False)
        
        # check if we're close enough that we consider the current sub goal fulfilled
        current_config = self.joints_sensor.joints_angles
        config_space_distance = np.linalg.norm(current_config - self.current_sub_goal)
        if config_space_distance < self.trajectory_tolerance:
            self.current_sub_goal = self.planned_trajectory.pop(0)

        
        # transform the current sub goal into a vector between -1 and 1, using the joint limits
        sub_goal_normalized = ((self.current_sub_goal - self.joints_limits_lower) * 2 / self.joints_range) - np.ones(6)
        # now we can use that to call the process_action method, which we use in joint mode (control mode 1 per the init of this class) to achieve the sub goal
        super().process_action(sub_goal_normalized)
        return process_time() - cpu_epoch
