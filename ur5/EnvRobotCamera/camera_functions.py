#%%
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from pybullet_util import MotionExecute
from typing import Union, List
from math_util import directionalVectorsFromQuaternion, add_list, getOrientationFromDirectionalVector

class CameraRail:
    
    
    def __init__(self, x_low, x_high, y_low, y_high, z_height, phi= 0): # phi in RAD
        x_mid = (x_high + x_low)/2
        y_mid = (y_high + y_low)/2
        self.center = np.array([x_mid, y_mid])
        self.radius = max((x_high - x_low)/2, (y_high - y_low)/2)
        self.z = z_height
        self.phi = phi
        self.position = self._get_coords()
        self.vel = 0


    def get_coords(self, d_phi, factor = 0.1):
        self.phi += np.clip(d_phi, -2*np.pi/50, 2*np.pi/50)
        
        return self._get_coords()

    def _get_coords(self) -> list:
        x = np.cos(self.phi) * self.radius
        y = np.sin(self.phi) * self.radius

        return [self.center[0] + x, self.center[1] + y, self.z]

class CameraRailRobotStraight:

    def __init__(self, center: Union[List, np.ndarray], center_offset_direction : Union[List, np.ndarray], center_offset_distance : float, length : float, z_height : float = 1.0, max_vel : float= 0.069):
        if type(center) is list: center = np.array(center)
        self.center = center
        if type(center_offset_direction) is list: center_offset_direction = np.array(center_offset_direction)
        self.codir : np.ndarray = center_offset_direction
        self.codir = self.codir/np.linalg.norm(self.codir)

        self.codist = center_offset_distance
        self.length = length
        self.pos_rel_to_length = 0
        self.max_vel = max_vel

        self.vel = 0
        self.position : np.ndarray = center + self.codist * self.codir
        self.position[2] = z_height
        self.vec = np.array([self.codir[1], -self.codir[0], 0])
        self.vec = self.vec / np.linalg.norm(self.vec)

    def _get_coords(self):
        self.vel = np.clip(self.vel, -self.max_vel, self.max_vel)

        self.pos_rel_to_length += self.vel
        if np.abs(self.pos_rel_to_length) > self.length/2:
            self.vel = 0
            self.pos_rel_to_length = np.sign(self.pos_rel_to_length) * self.length/2

        self.position += self.vel * self.vec


    def get_coords(self, d_vel):
        self.vel += d_vel
        self._get_coords()

        return self.position.tolist()

        





class CameraRailRobotCircle:
    
    
    def __init__(self, center, radius, z_height, phi_min= -np.pi, phi_max= np.pi, phi_offset = 0, x_y_offset= None, max_step_size= np.pi/25, phi= 0): # phi in RAD
        if type(center) is list: center = np.array(center)
        self.center = center
        self.radius = radius
        self.z = z_height
        self.phi = phi
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.phi_offset = phi_offset
        self.max_step_size = max_step_size
        self.x_y_offset = x_y_offset
        if self.x_y_offset is None:
            self.x_y_offset = [0, 0]
        self.position = self._get_coords()
        self.vel = 0


    def _clip_phi(self):
        self.phi = np.clip(self.phi, self.phi_min + self.phi_offset, self.phi_max + self.phi_offset)

    def get_coords(self, d_phi):
        self.phi += np.clip(d_phi, -self.max_step_size, self.max_step_size)
        
        return self._get_coords()

    def _get_coords(self) -> list:
        self._clip_phi()
        x = np.cos(self.phi) * self.radius
        y = np.sin(self.phi) * self.radius

        return [self.center[0] + x + self.x_y_offset[0], self.center[1] + y + self.x_y_offset[1], self.z]

class CameraRobot:
    """
    Camera being held by robot, range of motion is reduced by the range of the robot.

        :param urdf_root_path: path to urdf file of robot
        :param reference: Base of actor robot (where the camera will be looking at.)
        :param d_reference: Difference from point of reference to base of robot. (d_x, d_y, d_z)
        :param target: Camera target (e.g. middle of the workspace)
        :param fov: FOV of the camera
    """

    def __init__(self, urdf_root_path: str, reference: np.ndarray, d_reference: np.ndarray, target: np.ndarray, fov: int = 60, image_height= 128, image_width= 128, is_training= False, debug_parameters= False):
        self.base_pos = reference + d_reference
        self.base_orn = p.getQuaternionFromEuler([0,0,0])
        self.RobotUid = p.loadURDF(urdf_root_path, basePosition=self.base_pos, baseOrientation=self.base_orn, useFixedBase=True)
        self.base_link = 1
        self.effector_link = 7
        self.motionexec = MotionExecute(self.RobotUid, self.base_link, self.effector_link)
        self.init_home = reference + d_reference/2
        self.init_orn = p.getQuaternionFromEuler([0,0,0])  
        self.motionexec.go_to_target(self.init_home, self.init_orn)
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.target = target
        self.debug_parameters = debug_parameters
        if debug_parameters:
            self.orientation_debug = [p.addUserDebugParameter(f'roll', -10, 10, 0), p.addUserDebugParameter(f'pitch', -10, 10, 0), p.addUserDebugParameter(f'yaw', -10, 10, 0)]
            self.position_debug = [p.addUserDebugParameter(f'x', -5, 5, 0.7), p.addUserDebugParameter(f'y', -5, 5, 0.7), p.addUserDebugParameter(f'z', -2, 2, 0.4)]
            self.target_debug = [p.addUserDebugParameter(f'x_target', -5, 5, 0), p.addUserDebugParameter(f'y_target', -5, 5, 0.4), p.addUserDebugParameter(f'z_target', -2, 2, 0.3)]
        self.d_phi_debug = p.addUserDebugParameter(f'd_phi', -0.1, 0.1, 0)
        self.fov = fov
        self.image_height = image_height
        self.image_width = image_width
        # self.bahn = CameraRailRobotCircle(self.base_pos, radius= 0.5, z_height= 0.5, phi_min= np.pi, phi_max= 2*np.pi, phi_offset= 0, x_y_offset= [0, 0], phi= np.pi*3/2)
        self.bahn = CameraRailRobotStraight(self.base_pos, add_list(self.target, self.base_pos, -1), 0.5, 2, 0.2)
        self.is_training = is_training

        self.current_joint_position = None

        self.debug_lines = {}

    def update_current_joint_position(self):
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])   
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]

    def move_effector(self, d_position):
        if not self.is_training and self.debug_parameters:
            position = self.bahn.get_coords(p.readUserDebugParameter(self.d_phi_debug))
        else:
            position = self.bahn.get_coords(d_position)

        orientation = getOrientationFromDirectionalVector(add_list(self.target, position, -1))
        #orientation = [1,0,0,1]
        
        self.motionexec.go_to_target(position, p.getEulerFromQuaternion(orientation))
        self.update_current_joint_position()

    def _remove_debug_lines(self, line_ids):
        for line_id in line_ids:
            if self.debug_lines.get(line_id, None) is not None:
                p.removeUserDebugItem(self.debug_lines.get(line_id))

    def _set_camera(self, position, orientation, camera_type= 'rgb', debug_lines= True):
        
        self._remove_debug_lines(self.debug_lines.keys())

        # up_vector, forward_vector, left_vector = directionalVectorsFromQuaternion(orientation)
        
        if not self.is_training and self.debug_parameters:
            x = p.readUserDebugParameter(self.target_debug[0])
            y = p.readUserDebugParameter(self.target_debug[1])
            z = p.readUserDebugParameter(self.target_debug[2])
            target = [x, y, z]
            self.target = target
        else:
            target = self.target

        if not self.is_training and debug_lines:
            # self.debug_lines['forward'] = p.addUserDebugLine(position, add_list(position, forward_vector), [255, 0, 0])
            # self.debug_lines['left'] = p.addUserDebugLine(position, add_list(position, left_vector), [0, 255, 0])
            # self.debug_lines['up'] = p.addUserDebugLine(position, add_list(position, up_vector), [0,0,255])
            self.debug_lines['target'] = p.addUserDebugLine(position, target, [127, 127, 127])
            pass



        viewMatrix = p.computeViewMatrix(
            cameraTargetPosition=target,
            cameraEyePosition= position,
            cameraUpVector= [0, 0, 1],
            )

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov= self.fov,
            aspect=1,
            nearVal= 0.05,
            farVal= 4.0,
            )


        def _set_camera_inner(): # TODO           
            _, _, rgba, depth, _ = p.getCameraImage(
                width= self.image_width,
                height= self.image_height,
                viewMatrix= viewMatrix,
                projectionMatrix= projectionMatrix,
            )
            if camera_type == 'grayscale':
                r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]/255
                image = (0.2989 * r + 0.5870 * g + 0.1140 * b)*a
                image = image[None]
            if camera_type in ['rgb']:
                image = rgba.copy()[:, :, :3]
            if camera_type == 'rgbd':
                image = rgba.copy()
                image[:, :, 3] = depth


            return image

        
        return _set_camera_inner

    def get_image(self, target= None):

        return self._set_camera(self.current_pos, self.current_orn)()

    
    def move_camera(self, d_position, target=None):
        if target is not None:
            self.target = target
        self.move_effector(d_position)
        return self.get_image()






#%%
if __name__ == '__main__':
    pa = CameraRailRobotStraight([0,0,0], [0,1,0], 0.5, 0.5, 1)
    print(pa.vec)
# %%