import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from env import Env
import glob
from math_util import getOrientationFromDirectionalVector
from random import choice

def get_last_save(folder='./models/reach_ppo_ckp_logs', prefix= 'reach'):
    saves = os.listdir(folder)
    prev_steps = 0
    for save in saves:
        if not save.startswith(prefix): continue
        save : str
        candidate, ext = os.path.splitext(save)
        steps = int(candidate.split('_')[-2])
        if steps > prev_steps:
            chosen_file = candidate
            chosen_ext = ext
            prev_steps = steps

    return f'{folder}/{chosen_file}'

experiments = {
    'one' : {
        'obstacles': {
            'one' : {
                'position' : [0.0, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
        },
        'targets' : [[-0.15, 0.35, 0.2]],
        'start' : {
            'pos' : [0.15, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },      
    },
    'two' : {
        'obstacles': {
            'one' : {
                'position' : [-0.1, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'two' : {
                'position' : [0.1, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            }
        },
        'targets' : [[0.0, 0.35, 0.2], [-0.25, 0.35, 0.2]],
        'start' : {
            'pos' : [0.25, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },
         
    },
    'three' : {
        'obstacles': {
            'one' : {
                'position' : [-0.15, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'two' : {
                'position' : [0.0, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'three' : {
                'position' : [0.15, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
        },
        'targets' : [[0.075, 0.35, 0.2], [-0.075, 0.35, 0.2], [-0.3, 0.35, 0.2]],
        'start' : {
            'pos' : [0.3, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },
         
    },
    'one_reversed' : {
        'obstacles': {
            'one' : {
                'position' : [0.0, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
        },
        'targets' : [[0.15, 0.35, 0.2]],
        'start' : {
            'pos' : [-0.15, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },      
    },
    'two_reversed' : {
        'obstacles': {
            'one' : {
                'position' : [-0.1, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'two' : {
                'position' : [0.1, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            }
        },
        'targets' : [[0.0, 0.35, 0.2], [0.25, 0.35, 0.2]],
        'start' : {
            'pos' : [-0.25, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },
         
    },
    'three_reversed' : {
        'obstacles': {
            'one' : {
                'position' : [-0.15, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'two' : {
                'position' : [0.0, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'three' : {
                'position' : [0.15, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
        },
        'targets' : [[-0.075, 0.35, 0.2], [0.075, 0.35, 0.2], [0.3, 0.35, 0.2]],
        'start' : {
            'pos' : [-0.3, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },
         
    },
}

params = {
    'is_render': True, 
    'is_good_view': True,
    'is_train' : False,
    'show_boundary' : True,
    'add_moving_obstacle' : False,
    'moving_obstacle_speed' : 0.45,
    'moving_init_direction' : -1,
    'moving_init_axis' : 0,
    'workspace' : [-0.4, 0.4, 0.3, 0.7, 0.2, 0.4],
    'max_steps_one_episode' : 1024,
    'num_obstacles' : 15,
    'prob_obstacles' : 0.5,
    'obstacle_box_size' : [0.04,0.04,0.002],
    'obstacle_sphere_radius' : 0.04,
    'camera_args' : {
        'placement' : 'duo',
        'type' : 'rgb',
        'prev_pos' : 0,
        'visualize' : True,
        'follow_effector' : False,
    },
    'debug' : False,
    'experiment': {
        'obstacles': {
            'one' : {
                'position' : [-0.1, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
            'two' : {
                'position' : [0.1, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            }
        },
        'targets' : [[0.0, 0.35, 0.22], [-0.25, 0.35, 0.22]],
        'start' : {
            'pos' : [0.25, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },
    },
    'experiments' : experiments,
}



if __name__=='__main__':

    env = Env(
        is_render=params['is_render'],
        is_good_view=params['is_good_view'],
        is_train=params['is_train'],
        show_boundary=params['show_boundary'],
        add_moving_obstacle=params['add_moving_obstacle'],
        moving_obstacle_speed=params['moving_obstacle_speed'],
        moving_init_direction=params['moving_init_direction'],
        moving_init_axis=params['moving_init_axis'],
        workspace=params['workspace'],
        max_steps_one_episode=params['max_steps_one_episode'],
        num_obstacles=params['num_obstacles'],
        prob_obstacles=params['prob_obstacles'],
        obstacle_box_size=params['obstacle_box_size'],
        obstacle_sphere_radius=params['obstacle_sphere_radius'],
        camera_args=params['camera_args'],
        debug=params['debug'],
        #experiment=params['experiment'],
        # experiments=params['experiments'],
        )
    # load drl model
    # model = PPO.load('../../from_server/models/reach_ppo_ckp_logs/rgbd/reach_66355200_steps.zip', env=env) # Still had LIDAR and obs_space (C, 130, 128)
    # model = PPO.load(f'../../from_server/rgb/v5/reach_24000000_steps.zip', env=env)
    # model = PPO.load(f'from_server/rgb/v5/reach_24000000_steps.zip', env=env)
    model = PPO.load(f'../../from_server/rgb/experiments{"_follow_effector" if params["camera_args"]["follow_effector"] else ""}/reach_6000000_steps.zip', env=env)
    
    
    # top

    while True:
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
