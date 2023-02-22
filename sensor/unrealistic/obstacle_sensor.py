import pybullet as pyb
import numpy as np
from typing import Union, List, Dict, TypedDict
from robot.robot_implementations.ur5 import UR5
from gym import spaces
from robot import Robot
from ..sensor import Sensor

__all__ = [
    'ObstacleCenterRadius',
]

class ObstacleCenterRadius(Sensor):
    """
    Assumes that objects are not created/destroyed during experiment
    """

    def __init__(self, robot: Robot, max_obstacles: int, use_velocities : bool, name : str, add_to_observation_space : bool = True, add_to_logging : bool = False, update_steps : int = 1, n_stacked_frames:int = 1, **kwargs):
        self.robot = robot
        self.name = name
        self.out_name = 'obstacleradius_' + name
        self.max_obs = max_obstacles
        self.features_per_sphere = 7 if use_velocities else 4
        self.obstacles_state = np.zeros((max_obstacles, self.features_per_sphere))
        self.obstacles_pos_in_state = np.random.permutation(self.max_obs)
        self.stacked_obstacles_state = np.zeros((n_stacked_frames, max_obstacles, self.features_per_sphere))
        super().__init__(add_to_observation_space= add_to_observation_space, add_to_logging= add_to_logging, update_steps= update_steps, **kwargs)

    def update(self, step):
        for i, sphere in zip(self.obstacles_pos_in_state, self.robot.world.obstacle_objects):
            tmp = [*sphere.position, sphere.radius]
            if self.features_per_sphere > 4: tmp.extend([*sphere.velocity])
            self.obstacles_state[i] = np.array(tmp)
        self.stacked_obstacles_state = np.roll(self.stacked_obstacles_state, -1, axis= 0)
        self.stacked_obstacles_state[-1, :, :] = self.obstacles_state.copy()

    def reset(self):
        self.obstacles_state = np.zeros_like(self.obstacles_state)
        self.obstacles_pos_in_state = np.random.permutation(self.max_obs)
        self.stacked_obstacles_state = np.zeros_like(self.stacked_obstacles_state)

    def get_observation(self) -> dict:
        return {self.out_name : np.squeeze(self.stacked_obstacles_state)}

    def get_observation_space_element(self) -> dict:
        d = {}
        w = self.robot.world
        d[self.out_name] = spaces.Box(
            low = min(w.x_min, w.y_min, w.z_min, w.sphere_r_min), # TODO also consider sphere velocities
            high= max(w.x_max, w.y_max, w.z_max, w.sphere_r_max), # To be fair they are MOST LIKELY in this interval
            shape= np.squeeze(self.stacked_obstacles_state).shape, # Worst case we get some clipping/aliasing can't be that bad
        )
        return d

    def _normalize(self) -> dict:
        return self.get_observation()
