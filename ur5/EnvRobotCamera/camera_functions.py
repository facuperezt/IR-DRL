#%%
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from pybullet_util import MotionExecute
from math_util import directionalVectorsFromQuaternion, add_list

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

class CameraRobot:
    """
    Camera being held by robot, range of motion is reduced by the range of the robot.

        :param urdf_root_path: path to urdf file of robot
        :param reference: Base of actor robot (where the camera will be looking at.)
        :param d_reference: Difference from point of reference to base of robot. (d_x, d_y, d_z)
        :param target: Camera target (e.g. middle of the workspace)
        :param fov: FOV of the camera
    """

    def __init__(self, urdf_root_path: str, reference: np.ndarray, d_reference: np.ndarray, target: np.ndarray, fov: int = 90, image_height= 128, image_width= 128):
        self.base_pos = reference + d_reference
        self.base_orn = p.getQuaternionFromEuler(-d_reference)
        self.RobotUid = p.loadURDF(urdf_root_path, basePosition=self.base_pos, baseOrientation=self.base_orn, useFixedBase=True)
        self.base_link = 1
        self.effector_link = 7
        self.motionexec = MotionExecute(self.RobotUid, self.base_link, self.effector_link)
        self.init_home = reference + d_reference/2
        self.init_orn = p.quaternionFromEuler([0,0,0])
        self.motionexec.go_to_target(self.init_home, self.init_orn)
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.target = target
        self.orientation_debug = [p.addUserDebugParameter(f'{i}o') for i in [1, 2, 3]]
        self.position_debug = [p.addUserDebugParameter(f'{i}p') for i in [1, 2, 3]]
        self.fov = fov
        self.image_height = image_height
        self.image_width = image_width

        self.current_joint_position = None

        self.debug_lines = {}

    def get_current_joint_position(self):
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])    

    def move_effector(self, position, orientation):
        self.motionexec.go_to_target(position, orientation)
        self._set_camera()

    def _remove_debug_lines(self, line_ids):
        for line_id in line_ids:
            if self.debug_lines.get(line_id, None) is not None:
                p.removeUserDebugItem(self.debug_lines.get(line_id))

    def _set_camera(self, position = None, orientation = None, camera_type= 'rgb', debug_lines= True):
        
        self._remove_debug_lines(self.debug_lines.keys())

        if orientation is None:
            orientation = [p.readUserDebugParameter(p) for p in self.orientation_debug]
        if position is None:
            position = [p.readUserDebugParameter(p) for p in self.position_debug]

        up_vector, forward_vector, left_vector = directionalVectorsFromQuaternion(orientation)

        if debug_lines:
            self.debug_lines['forward'] = p.addUserDebugLine(position, add_list(position, forward_vector, [255, 0, 0]))
            self.debug_lines['left'] = p.addUserDebugLine(position, add_list(position, left_vector, [0, 255, 0]))
            self.debug_lines['up'] = p.addUserDebugLine(position, add_list(position, up_vector), [0,0,255])

        target = self.target


        self.move_effector(position)

        viewMatrix = p.computeViewMatrix(
            cameraTargetPosition=target,
            cameraEyePosition= position,
            cameraUpVector= up_vector,
            )

        projectionMatrix = p.computeProjectionMatrixFOV(
            # left=self.x_low_obs,
            # right=self.x_high_obs,
            # top=self.z_high_obs,
            # bottom=self.z_low_obs,
            # nearVal=self.y_high_obs,
            # farVal=self.y_low_obs,
            fov= self.fov,
            aspect=1,
            nearVal= 0.1,
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

    def get_image(self, position= None, orientation = None):

        return self._set_camera(position, orientation)()





#%%
if __name__ == '__main__':

    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi, 100)

    # the radius of the circle
    r = np.sqrt(1)

    # compute x1 and x2
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)

    # create the figure
    fig, ax = plt.subplots(1)
    ax.plot(x1, x2)

    c = CameraRail(-1, 1, -0.1, 0.1, 0)
    point = c.get_coords(30)[:2]

    ax.plot(*point, 'x')

    ax.set_aspect(1)
    plt.show()
# %%
