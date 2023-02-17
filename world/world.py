from abc import ABC, abstractmethod
import numpy as np
import pybullet as pyb

class World(ABC):
    """
    Abstract Base Class for a simulation world. Methods signed with abstractmethod need to be implemented by subclasses.
    See the random obstacles world for examples.
    """

    def __init__(self, workspace_boundaries:list, sim_step: float, env_id: int):

        # list that will contain all PyBullet object ids with collision managed by this world simulation
        self.objects_ids = []
        # list that will contain all purely visual PyBullet object ids (e.g. explicatory lines, workspace boundaries etc.)
        self.aux_object_ids = []
        # list that will contain all obstacle objects used in this world
        self.obstacle_objects = []

        # set sim step
        self.sim_step = sim_step

        # set env id, this is a number that differentiates multiple envs running in parallel in training, necessary for file related stuff
        self.env_id = env_id

        # set up workspace boundaries
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = workspace_boundaries

        # targets for goals that need to interact with the world
        self.position_targets = []
        self.rotation_targets = []

        # points for robot end effectors at episode start
        self.ee_starting_points = []

        # list of robots, gets filled by register method down below
        self.robots_in_world = []  # all robots in world

        # collision attribute, for convenient outside access
        self.collision = False

    def register_robots(self, robots):
        """
        This method receives a list of robot objects from the outside and sorts the robots therein into several lists that are important for
        other methods.
        Also gives each robot an id int that can be used to identify it.
        """
        id_counter = 0
        for robot in robots:
            self.robots_in_world.append(robot)
            robot.id = id_counter
            id_counter += 1

    def perform_collision_check(self):
        """
        Performs a collision check 
        1. between all robots and all obstacles in the world and
        2. between each robot
        
        Stores the result in a class variable.
        """
        pyb.performCollisionDetection()
        col = False
        # check for each robot with every obstacle
        for robot in self.robots_in_world:
            for obj in self.objects_ids:
                if len(pyb.getContactPoints(robot.object_id, obj)) > 0:
                    col = True 
                    break
            if col:
                break  # this is to immediately break out of the outer loop too once a collision has been found
        # check for each robot with every other one
        if not col:  # skip if another collision was already detected
            for idx, robot in enumerate(self.robots_in_world[:-1]):
                for other_robot in self.robots_in_world[idx+1:]:
                    if len(pyb.getContactPoints(robot.object_id, other_robot.object_id)) > 0:
                        col = True
                        break
                if col:
                    break  # same as above
        self.collision = col

    @abstractmethod
    def build(self):
        """
        This method should build all the components that make up the world simulation aside from the robot.
        This includes URDF files as well as objects created by PyBullet code.
        All object ids loaded in by this method must be added to the self.object_ids list! Otherwise they will be ignored in collision detection.
        """
        pass

    @abstractmethod
    def reset(self, success_rate):
        """
        This method should reset all lists, arrays, variables etc. that handle the world to such a state that a new episode can be run.
        Meaning that after this method is done, build() can be called again.
        Don't reset the PyBullet simulation itself, that will be handled by the gym env.
        Additionally, this method receives the success rate of the gym env as a value between 0 and 1. You could use this to
        set certain parameters, e.g. to the make world become more complex as the agent's success rate increases.
        """
        pass

    def build_visual_aux(self):
        """
        This method should add objects that are not necessary to the purpose of the world and useful only for visual quality.
        By default, this method will mark the workspace boundaries, but you can extend it in your subclasses as you need.
        Visual objects related to a goal should be implemented by that goal.
        (This is because the world should be usable with all sorts of goals, even those that need different visualizations for their goals,
        e.g. a target sphere vs. a target cube)
        Objects built here should NOT be added to self.object_ids but to self.aux_object_ids. Dont forget to reset self.aux_object_ids in your reset methods.
        """
        a = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                                lineToXYZ=[self.x_min, self.y_min, self.z_max])
        b = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        c = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        d = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])

        e = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        f = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        g = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        h = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        
        i = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_min])
        j = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
        k = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_min])
        l = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])

        self.aux_object_ids += [a, b, c, d, e, f, g, h, i, j, k , l]
    
    @abstractmethod
    def update(self):
        """
        This method should update all dynamic and movable parts of the world simulation. If there are none it doesn't need to do anything at all.
        """
        pass

    @abstractmethod
    def create_ee_starting_points(self) -> list:
        """
        This method should return a valid starting position for the end effector at episode start.
        Valid meaning reachable and not in collision.
        The return should be a list of tuples, each containing a 3D Point and a quaternion (which can be None instead if no specific rotation is needed), one tuple for each robot registered in the world.
        """
        pass

    @abstractmethod
    def create_position_target(self) -> list:
        """
        This method should return a valid target position within the world simulation for a robot end effector.
        Valid meaning (at least very likely) being reachable for the robot without collision.
        The return value should be a list of 3D points as numpy arrays, one each for every robot registered with the world (robots that don't have the position goal should still get an empty entry).
        """
        pass

    @abstractmethod
    def create_rotation_target(self) -> list:
        """
        This method should return a valid target rotation within the world simulation for a robot end effector a
        Valid meaning (at least very likely) being reachable for the robot without collision.
        The return value should be a list of quaternions as numpy arrays, one for each robot registered with the world (robots that don't have the rotation goal should still get an empty entry).
        """
        pass
