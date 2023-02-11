from typing import Union
import numpy as np
import pybullet as pyb
from robot.robot import Robot

__all__ = [
    'UR5',
    'UR5_P2P',
]

class UR5(Robot):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta: float, rpy_delta: float):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, xyz_delta, rpy_delta)
        self.joints_limits_lower = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        self.joints_limits_upper = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.array([10., 10., 10., 10., 10., 10.])

        self.end_effector_link_id = 7
        self.base_link_id = 1

    def get_action_space_dims(self):
        return (6,6)  # 6 joints

    def build(self):

        self.object_id = pyb.loadURDF("robots/predefined/ur5/urdf/ur5.urdf", basePosition=self.base_position.tolist(), baseOrientation=self.base_orientation.tolist(), useFixedBase=True, flags=pyb.URDF_USE_SELF_COLLISION)
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