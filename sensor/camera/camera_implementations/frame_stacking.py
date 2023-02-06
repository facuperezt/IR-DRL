import pybullet as pyb
from typing import Union, List, Dict, TypedDict
from robot.robot_implementations.ur5 import UR5
from gym import spaces
from ..camera_utils import *
from ..camera import CameraBase, CameraArgs # to prevent circular imports the things within the package have to be imported using the relative path


__all__ = [
    'StaticFrameStacking',
]

class StaticFrameStacking(CameraBase):
    """
    :)
    """

    def __init__(self, robot : UR5, position: List, target: List = None, n_frames = 5, camera_args : CameraArgs = None, name : str = 'default_floating', **kwargs):
        super().__init__(target= target, camera_args= camera_args, name= name, **kwargs)
        self.robot = robot
        self.pos = position
        self.n_frames = n_frames
        nr_channels = {
            'grayscale' : 1,
            'rgb' : 3,
            'rgbd': 4,
        }
        self.frames_buffer = torch.zeros((n_frames, self.camera_args["width"], self.camera_args["height"], nr_channels[self.camera_args["type"]]))

    def _adapt_to_environment(self):
        self.target = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4]
        super()._adapt_to_environment()

    def _set_camera(self):
        if self.debug.get('position', False) or self.debug.get('orientation', False) or self.debug.get('target', False) or self.debug.get('lines', False):
            self._use_debug_params()

        viewMatrix = pyb.computeViewMatrix(
            cameraTargetPosition=self.target,
            cameraEyePosition= self.pos,
            cameraUpVector= self.camera_args['up_vector'],
            )

        projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov= self.camera_args['fov'],
            aspect=self.camera_args['aspect'],
            nearVal= self.camera_args['near_val'],
            farVal= self.camera_args['far_val'],
            )


        def _set_camera_inner(): # TODO           
            _, _, rgba, depth, _ = pyb.getCameraImage(
                width= self.camera_args['width'],
                height= self.camera_args['height'],
                viewMatrix= viewMatrix,
                projectionMatrix= projectionMatrix,
            )
            rgba, depth = np.array(rgba), np.array(depth) # for compatibility with older python versions
            if self.camera_args['type'] == 'grayscale':
                r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]/255
                image = (0.2989 * r + 0.5870 * g + 0.1140 * b)*a
                image = image[None]
            if self.camera_args['type'] in ['rgb']:
                image = rgba[:, :, :3]
            if self.camera_args['type'] == 'rgbd':
                image = rgba
                image[:, :, 3] = depth


            return image

        self.camera_ready = True
        return _set_camera_inner

    def get_observation_space_element(self) -> Dict:
        nr_channels = {
            'grayscale' : 1,
            'rgb' : 3,
            'rgbd': 4,
        }
        low = 0
        high = 1 if self.normalize else 255
        return {self.output_name : spaces.Box(low=low, high= high, shape=(self.n_frames, self.camera_args['height'],self.camera_args['width'],nr_channels[self.camera_args['type']],), dtype=np.uint8),}

    def get_data_for_logging(self) -> dict:
        """
        Track target because reasons
        """
        dic = super().get_data_for_logging()
        dic[self.output_name + '_target'] = self.target
        dic[self.output_name + '_n_stacked_frames'] = self.n_frames
        return dic

    def reset(self):
        super().reset()
        self.frames_buffer = np.repeat(self.current_image[np.newaxis], self.n_frames, 0)

    def update(self, step):
        super().update(step)
        if (self.current_image != self.frames_buffer[-1]).all():
            self.frames_buffer = np.roll(self.frames_buffer, -1, 0)
            self.frames_buffer[-1] = self.current_image

        return self.get_observation()

    def get_observation(self):
        return {self.output_name : self.frames_buffer}
