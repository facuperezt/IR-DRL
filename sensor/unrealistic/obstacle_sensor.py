import pybullet as pyb
import numpy as np
from typing import Union, List, Dict, TypedDict
from robot.robot_implementations.ur5 import UR5
from gym import spaces
from robot import Robot
from ..sensor import Sensor
from world.obstacles.pybullet_shapes import Sphere

__all__ = [
    'ObstacleCenterRadius',
]

class ObstacleCenterRadius(Sensor):
    """
    Assumes that objects are not created/destroyed during experiment
    """

    def __init__(self, robot: Robot, max_obstacles: int, use_velocities : bool, use_importance : bool, name : str, add_to_observation_space : bool = True, add_to_logging : bool = False, update_steps : int = 1, n_stacked_frames:int = 1, **kwargs):
        self.robot = robot
        self.name = name
        self.out_name = 'obstacleradius_' + name
        self.max_obs = max_obstacles
        self.use_velocities = use_velocities
        self.use_importance = use_importance
        self.features_per_sphere = 4
        self.features_per_sphere += 3 if use_velocities else 0
        self.features_per_sphere += 1 if use_importance else 0
        self.obstacles_state = np.zeros((max_obstacles, self.features_per_sphere))
        self.obstacles_pos_in_state = np.random.permutation(self.max_obs)
        self.stacked_obstacles_state = np.zeros((n_stacked_frames, max_obstacles, self.features_per_sphere))
        super().__init__(add_to_observation_space= add_to_observation_space, add_to_logging= add_to_logging, update_steps= update_steps, **kwargs)


    def LinePlaneCollision(self, planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

        ndotu = planeNormal.dot(rayDirection)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi

    def get_closest_point_to_obstacles(self,):
        min_dists = []
        importances = []
        # pyb.removeAllUserDebugItems()
        for i, sphere in zip(self.obstacles_pos_in_state, self.robot.world.obstacle_objects):
            min_dist = np.inf
            sphere : Sphere
            closest_joint = [link for link in sorted(pyb.getClosestPoints(self.robot.object_id, sphere.object_id, 2), key= lambda x: x[8]) if link[3] != self.robot.base_link_id][:1] # closest joint
            if len(closest_joint) > 0:
                min_dist = closest_joint[0][8]
                _closest_point_to_obst = closest_joint[0][5]
                importance = sphere.importance if self.use_importance else 1
            else:
                min_dist = 2
                _closest_point_to_obst = [0,0,0]
                importance = 1
            # for start, end in zip(links[:-1], links[1:]):
            #     # pyb.addUserDebugLine(start[4], end[4])
            #     start = np.asarray(start[4])
            #     end = np.asarray(end[4])
            #     vec = end-start
            #     closest_point = self.LinePlaneCollision(vec, pos, vec, start)
            #     dist = np.clip(np.linalg.norm(pos - closest_point) - sphere.radius, a_min= 0, a_max= None)
            #     if dist < min_dist:
            #         min_dist = dist
            #         importance = sphere.importance
            #         _closest_point_to_obst = closest_point
            # pyb.addUserDebugLine(sphere.position, _closest_point_to_obst)
            min_dists.append(min_dist)
            importances.append(importance)
        return min_dists, importances


    def update(self, step):
        for i, sphere in zip(self.obstacles_pos_in_state, self.robot.world.obstacle_objects):
            tmp = [*sphere.position, sphere.radius]
            if self.use_velocities: tmp.extend([*sphere.velocity])
            if self.use_importance: tmp.extend([sphere.importance])
            self.obstacles_state[i] = np.array(tmp)
        self.stacked_obstacles_state = np.roll(self.stacked_obstacles_state, -1, axis= 0)
        self.stacked_obstacles_state[-1, :, :] = self.obstacles_state.copy()
        self.robot.goal.obstacles_info = self.get_closest_point_to_obstacles()

    def reset(self):
        self.obstacles_state = np.zeros_like(self.obstacles_state)
        self.obstacles_pos_in_state = np.random.permutation(self.max_obs)
        self.stacked_obstacles_state = np.zeros_like(self.stacked_obstacles_state)
        self.robot.goal.obstacles_info = self.get_closest_point_to_obstacles()

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
