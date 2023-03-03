from world.world import World
from world.obstacles.pybullet_shapes import Box, Sphere
import numpy as np
import pybullet as pyb
import pybullet_data as pyb_d
from random import choice, shuffle
import matplotlib.pyplot as plt
import matplotlib.colors

__all__ = [
    'TableWorld'
]

class TableWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed between the p
    Depending on the configuration, some of these can be moving in various directions at various speeds
    """
    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       env_id: int,
                       num_static_obstacles: int=3, 
                       num_moving_obstacles: int=1,
                       box_measurements: list=[0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125],
                       sphere_measurements: list=[0.005, 0.02],
                       moving_obstacles_vels: list=[0.5, 2],
                       moving_obstacles_directions: list=[],
                       moving_obstacles_trajectory_length: list=[0.05, 0.75],
                       fixed_nr_obst: bool = False,
                       ):
        """
        The world config contains the following parameters:
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param num_static_obstacles: int number that is the amount of static obstacles in the world
        :param num_moving_obstacles: int number that is the amount of moving obstacles in the world
        :param sim_step: float for the time per sim step
        :param box_measurements: List of 6 floats that gives the minimum and maximum dimensions of box shapes in the following order: lmin, lmax, wmin, wmax, hmin, hmax
        :param sphere_measurements: List of 2 floats that gives the minimum and maximum radius of sphere shapes
        :param moving_obstacles_vels: List of 2 floats that gives the minimum and maximum velocity dynamic obstacles can move with
        :param moving_obstacles_directions: List of numpy arrays that contain directions in 3D space among which obstacles can move. If none are given directions are generated in random fashion.
        :param moving_obstacles_trajectory_length: List of 2 floats that contains the minimum and maximum trajectory length of dynamic obstacles.
        """
        # TODO: add random rotations for the plates

        super().__init__(workspace_boundaries, sim_step, env_id)

        self.fixed_nr_obst = fixed_nr_obst

        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles

        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, self.box_h_min, self.box_h_max = box_measurements
        self.sphere_r_min, self.sphere_r_max = sphere_measurements

        self.vel_min, self.vel_max = moving_obstacles_vels

        self.allowed_directions = [np.array(direction) for direction in moving_obstacles_directions]

        self.trajectory_length_min, self.trajectory_length_max = moving_obstacles_trajectory_length

        self.obstacle_objects : Sphere = []  # list to access the obstacle python objects

        self.ground_and_table_id = []
        # ground plate
        self.ground_and_table_id.append(pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01]))
        # table
        self.ground_and_table_id.append(pyb.loadURDF(pyb_d.getDataPath()+"/table/table.urdf", useFixedBase=True, globalScaling=1.75))

    def LinePlaneCollision(self, planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

        ndotu = planeNormal.dot(rayDirection)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi

    def valid_sphere_position(self, sphere : Sphere, points : list = None, safety_radii : list = None):
        if points is None:
            points = [[0,0,0.5]]
        if safety_radii is None:
            safety_radii = [[0.2]]
        assert len(points) == len(safety_radii), 'Each point needs a safety radius.'
        plane_foo = lambda vec, off: lambda xyz: np.dot(vec, xyz) + off
        for point, rp in zip(points, safety_radii):
            p0 = np.asarray(point)
            rp = np.asarray(rp)
            if np.linalg.norm(sphere.position - p0) < rp + sphere.radius: return False
            if len(sphere.trajectory) > 0:
                if any([np.linalg.norm(trajectory_stop - point) < rp + sphere.radius for trajectory_stop in sphere.trajectory[1:]]): return False
                for p1, p2 in zip(sphere.trajectory[:-1], sphere.trajectory[1:]):
                    p1, p2 = np.asarray(p1), np.asarray(p2)
                    vec = p2-p1
                    closest_point = self.LinePlaneCollision(vec, p0, vec, p2)
                    if np.linalg.norm(p0 - closest_point) < rp + sphere.radius: return False
        return True


    def build(self):
        random_object_amount = np.random.rand() if not self.fixed_nr_obst else 1
        cvals  = [0., 0.15,  1]
        colors = ["red", "salmon", "dimgray"]

        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        # add the moving obstacles
        while len(self.obstacle_objects)  < int(random_object_amount * (self.num_moving_obstacles + self.num_static_obstacles)):
            
            position = np.random.uniform(low=(self.x_min, self.y_min, self.z_min), high=(self.x_max, self.y_max, self.z_max), size=(3,))

            # moving obstacles
            if len(self.obstacle_objects) < int(random_object_amount *  self.num_moving_obstacles):
                # generate a velocity
                move_step = np.random.uniform(low=self.vel_min, high=self.vel_max) * self.sim_step
                # generate a trajectory length
                trajectory_length = np.random.uniform(low=self.trajectory_length_min, high=self.trajectory_length_max)
                # get the direction from __init__ or, if none are given, generate one at random
                if self.allowed_directions:
                    direction = choice(self.allowed_directions)
                else:
                    direction = np.random.uniform(low=-1, high=1, size=(3,))
                direction = (trajectory_length / np.linalg.norm(direction)) * direction
                goal_for_movement = direction + position
                trajectory = [position, goal_for_movement]  # loop between two points
            # static ones
            else:
                move_step = 0
                trajectory = []
            # sphere
            # generate random size
            radius = np.random.uniform(low=self.sphere_r_min, high=self.sphere_r_max)
            importance = np.random.uniform(0.05, 1)
            sphere = Sphere(position, [0, 0, 0, 1], trajectory, move_step, radius, importance= importance, color=cmap(importance))
            if not self.valid_sphere_position(sphere,): continue
            self.obstacle_objects.append(sphere)
            self.objects_ids.append(sphere.build())

            pyb.performCollisionDetection()
            for robot in self.robots_in_world:
                if len(pyb.getContactPoints(robot.object_id, self.objects_ids[-1])) > 0:
                    del self.obstacle_objects[-1]
                    pyb.removeBody(self.objects_ids[-1])
                    del self.objects_ids[-1]



    def reset(self, success_rate):
        self.objects_ids = self.ground_and_table_id
        self.starting_joint_states = []
        self.target_joint_states = []
        for object in self.obstacle_objects:
            pyb.removeBody(object.object_id)
            del object
        self.obstacle_objects = []
        # the next three don't need to be reset, so commented out
        #self.robots_in_world = []
        #self.robots_with_position = []
        #self.robots_with_orientation = []

    def update(self):

        for obstacle in self.obstacle_objects:
            obstacle.move()
    
    def create_starting_states(self):
        return self.create_starting_joint_states()

    def create_target_states(self):
        return self.create_target_joint_states()


    def create_starting_joint_states(self):
        for robot in self.robots_in_world:
            if robot.goal.needs_a_position: # re-use the old flag
                self.starting_joint_states.append(robot._get_random_valid_angles())
            else:
                self.starting_joint_states.append([])

        return self.starting_joint_states
    
    def create_target_joint_states(self):
        starting_joint_positions = []
        for robot in self.robots_in_world:
            starting_joint_positions.append([a[0] for a in pyb.getJointStates(robot.object_id, robot.joints_ids)])

        while True:
            tmp_target_joint_states = []
            for robot in self.robots_in_world:
                if robot.goal.needs_a_position:
                    candidate_joint_positions = robot._get_random_valid_angles()
                    robot.moveto_joints(candidate_joint_positions, False)
                    tmp_target_joint_states.append(candidate_joint_positions)
                else:
                    tmp_target_joint_states.append([])
            self.perform_collision_check()
            if not self.collision:
                self.collision = False
                self.target_joint_states = tmp_target_joint_states
                for robot, joint_positions in zip(self.robots_in_world, starting_joint_positions):
                    robot.moveto_joints(joint_positions, False)
                break
        return self.target_joint_states

    def setup_starting_states(self):
        for robot, state in zip(self.robots_in_world, self.starting_joint_states):
            robot.moveto_joints(state, False)

    def create_ee_starting_points(self) -> list:
        raise ValueError
        return super().create_ee_starting_points()

    def create_position_target(self) -> list:
        raise ValueError
        return super().create_position_target()    

    def create_rotation_target(self) -> list:
        raise ValueError
        return super().create_rotation_target()

    def build_visual_aux(self):
                # create a visual border for the workspace
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                                lineToXYZ=[self.x_min, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])

        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
