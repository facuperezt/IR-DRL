from goal.goal import Goal
import numpy as np
from robot.robot import Robot
from gym.spaces import Box
import pybullet as pyb
from functools import reduce
from sensor import JointsSensor

__all__ = [
    'JointsCollisionGoal',
]

class JointsCollisionGoal(Goal):

    def __init__(self, robot: Robot, 
                       normalize_rewards: bool, 
                       normalize_observations: bool,
                       train: bool,
                       add_to_logging: bool,
                       max_steps: int,
                       continue_after_success:bool, 
                       reward_success=10, 
                       reward_collision=-10,
                       reward_distance_mult=-0.01,
                       dist_threshold_start=3e-1,
                       dist_threshold_end=1e-2,
                       dist_threshold_increment_start=1e-2,
                       dist_threshold_increment_end=1e-3,
                       dist_threshold_overwrite:float=None):

        super().__init__(robot, normalize_rewards, normalize_observations, train, True, add_to_logging, max_steps, continue_after_success)

        # set output name for observation space
        self.output_name = "PositionGoal_" + self.robot.name

        # set the flags
        self.needs_a_position = True
        self.needs_a_rotation = False

        # set the reward that's given if the ee reaches the goal position and for collision
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        
        # multiplicator for the distance reward
        self.reward_distance_mult = reward_distance_mult

        # set the distance thresholds and the increments for changing them
        self.distance_threshold = dist_threshold_start if self.train else dist_threshold_end
        if dist_threshold_overwrite:  # allows to set a different startpoint from the outside
            self.distance_threshold = dist_threshold_overwrite
        self.distance_threshold_start = dist_threshold_start
        self.distance_threshold_end = dist_threshold_end
        self.distance_threshold_increment_start = dist_threshold_increment_start
        self.distance_threshold_increment_end = dist_threshold_increment_end

        # set up normalizing constants for faster normalizing
        #     reward
        max_reward_value = max(abs(self.reward_success), abs(self.reward_collision))
        self.normalizing_constant_a_reward = 2 / (2 * max_reward_value)
        self.normalizing_constant_b_reward = 1 - self.normalizing_constant_a_reward * max_reward_value
        #     observation
        #       get maximum ranges from world associated with robot
        j : JointsSensor = self.robot.joints_sensor
        vec_distance_max = self.robot.joints_limits_upper - self.robot.joints_limits_lower
        vec_distance_min = -1 * vec_distance_max
        distance_max = np.linalg.norm(vec_distance_max)
        #       constants
        self.normalizing_constant_a_obs = np.zeros(j.joints_dims + 1)  # joints_dims for difference vector and 1 for distance itself
        self.normalizing_constant_b_obs = np.zeros(j.joints_dims + 1)  # joints_dims for difference vector and 1 for distance itself
        self.normalizing_constant_a_obs[:j.joints_dims] = 2 / (vec_distance_max - vec_distance_min)
        self.normalizing_constant_a_obs[j.joints_dims] = 1 / distance_max  # distance only between 0 and 1
        self.normalizing_constant_b_obs[:j.joints_dims] = np.ones(j.joints_dims) - np.multiply(self.normalizing_constant_a_obs[:j.joints_dims], vec_distance_max)
        self.normalizing_constant_b_obs[j.joints_dims] = 1 - self.normalizing_constant_a_obs[3] * distance_max  # this is 0, but keeping it in the code for symmetry

        self.distance = None
        self.position = None
        self.joints_angles = j.joints_angles
        self.reward_value = 0
        self.shaking = 0 # needed?
        self.collided = False
        self.timeout = False
        self.out_of_bounds = False # needed?
        self.is_success = False
        self.done = False
        self.past_joints_angles = []

        # performance metric name
        self.metric_name = "angle_dist_threshold"

        self.goal_vis = None

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            if self.normalize_observations:
                ret[self.output_name ] = Box(low=-1, high=1, shape=(7,), dtype=np.float32)
            else:
                high = np.pi
                low = -np.pi
                ret[self.output_name ] = Box(low=low, high=high, shape=(7,), dtype=np.float32)

            return ret
        else:
            return {}

    def get_observation(self) -> dict:
        # get the data
        self.position = self.robot.position_rotation_sensor.position
        self.joints = self.robot.joints_sensor.joints_angles
        self.target = self.robot.world.target_joint_states[self.robot.id].copy()
        dif = self.target - self.joints
        self.distance = np.linalg.norm(dif)

        self.past_joints_angles.append(self.joints)
        if len(self.past_joints_angles) > 10:
            self.past_joints_angles.pop(0)

        ret = np.zeros(len(self.joints) + 1)
        ret[:len(self.joints)] = self.target
        ret[len(self.joints)] = self.distance
        
        if self.normalize_observations:
            return {self.output_name: np.multiply(self.normalizing_constant_a_obs, ret) + self.normalizing_constant_b_obs} 
        else:
            return {self.output_name: ret}

    def _compare_pose_similarity(self, i):
        return int(np.linalg.norm(self.past_joints_angles[i+1]-self.past_joints_angles[i]) < np.linalg.norm(self.past_joints_angles[i+1] - self.past_joints_angles[i-1]))

    def reward(self, step, action):
        
        reward = 0

        self.out_of_bounds = self._out()
        self.collided = self.robot.world.collision

        # shaking = 0
        # if len(self.past_joints_angles) >= 10:
        #     for i in range(1,len(self.past_joints_angles) - 1):
        #         shaking += self._compare_pose_similarity(i)
        # self.shaking = shaking
        # reward -= shaking / len(self.past_joints_angles) - 2
        
        # reward -= np.sum(np.array(action)**2 / (len(action)/2))

        # if len(self.past_joints_angles) >= 2:
        #     reward += 1 if np.linalg.norm(self.past_joints_angles[-1] - self.target) < np.linalg.norm(self.past_joints_angles[-2] - self.target) else 1


        self.is_success = False
        # reward += (np.abs(self.joints - self.target) < self.distance_threshold).sum() * 0.1
        if self.collided:
            self.done = True
            reward += self.reward_collision
        elif (np.abs(self.joints - self.target) < self.distance_threshold).all():
            self.done = True
            self.is_success = True
            reward += self.reward_success
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            # reward += self.reward_collision / 2
        else:
            self.done = False
            reward += self.reward_distance_mult * self.distance
        

        
        self.reward_value = reward
        if self.normalize_rewards:
            self.reward_value = self.normalizing_constant_a_reward * self.reward_value + self.normalizing_constant_b_reward

        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds
    
    def on_env_reset(self, success_rate):
        
        self.timeout = False
        self.is_success = False
        self.is_done = False
        self.collided = False
        self.out_of_bounds = False
        
        # set the distance threshold according to the success of the training
        if True or self.train: 

            # calculate increment
            ratio_start_end = (self.distance_threshold - self.distance_threshold_end) / (self.distance_threshold_start - self.distance_threshold_end)
            increment = (self.distance_threshold_increment_start - self.distance_threshold_increment_end) * ratio_start_end + self.distance_threshold_increment_end
            if success_rate > 0.8 and self.distance_threshold > self.distance_threshold_end:
                if success_rate == 1 and self.distance_threshold < self.distance_threshold_end * 5: self.robot.world.num_static_obstacles += 1
                self.distance_threshold -= increment 
            elif success_rate < 0.2 and self.distance_threshold < self.distance_threshold_start:
                if success_rate == 0 and self.distance_threshold < self.distance_threshold_increment_end * 5: self.robot.world.num_static_obstacles -= 1
                self.distance_threshold += increment  # upwards movement should be slower
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end

        return self.metric_name, self.distance_threshold, True, True

    def build_visual_aux(self):
        if self.goal_vis is None:
            self.goal_vis = pyb.loadURDF("robots/predefined/ur5/urdf/ur5_no_collision.urdf", basePosition=self.robot.base_position.tolist(), baseOrientation=self.robot.base_orientation.tolist(), useFixedBase=True, globalScaling=1)
            self.joints_info = [pyb.getJointInfo(self.goal_vis, i) for i in range(pyb.getNumJoints(self.goal_vis))]
            self.joints_ids = np.array([j[0] for j in self.joints_info if j[2] == pyb.JOINT_REVOLUTE])
        target = self.robot.world.target_joint_states[self.robot.id].copy()
        for i in range(len(self.joints_ids)):
            pyb.resetJointState(self.goal_vis, self.joints_ids[i], target[i])
        
        
        
        # # build a sphere of distance_threshold size around the target
        # self.target = self.robot.world.position_targets[self.robot.object_id]
        # self.goal_vis = pyb.createMultiBody(baseMass=0,
        #                     baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.distance_threshold, rgbaColor=[0, 1, 0, 1]),
        #                     basePosition=self.target)
        pass

    def get_data_for_logging(self) -> dict:
        logging_dict = {}

        logging_dict["shaking_" + self.robot.name] = self.shaking
        logging_dict["reward_" + self.robot.name] = self.reward_value
        logging_dict["distance_" + self.robot.name] = self.distance
        logging_dict["distance_threshold_" + self.robot.name] = self.distance_threshold

        return logging_dict

    ###################
    # utility methods #
    ###################

    def _out(self):
        
        x, y, z = self.position
        if x > self.robot.world.x_max or x < self.robot.world.x_min:
            return True
        elif y > self.robot.world.y_max or y < self.robot.world.y_min:
            return True
        elif z > self.robot.world.z_max or z < self.robot.world.z_min:
            return True
        return False