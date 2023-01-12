#%%
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
import tensorboard
from custom_policy import CustomCombinedExtractor
import argparse

CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from env import Env

experiments = {
    'one' : {
        'obstacles': {
            'one' : {
                'position' : [0.0, 0.4, 0.25],
                'orientation' : [0, 0.707, 0, 0.707],
                'size' : [0.07, 0.10, 0.002],
            },
        },
        'targets' : [[-0.15, 0.35, 0.21]],
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
        'targets' : [[0.0, 0.35, 0.21], [-0.25, 0.35, 0.21]],
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
        'targets' : [[0.075, 0.35, 0.21], [-0.075, 0.35, 0.21], [-0.3, 0.35, 0.21]],
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
        'targets' : [[0.15, 0.35, 0.21]],
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
        'targets' : [[0.0, 0.35, 0.21], [0.25, 0.35, 0.21]],
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
        'targets' : [[-0.075, 0.35, 0.21], [0.075, 0.35, 0.21], [0.3, 0.35, 0.21]],
        'start' : {
            'pos' : [-0.3, 0.35, 0.3],
            'orn' : [np.pi,0,np.pi],
        },
         
    },
}

def get_last_save(folder='./models/reach_ppo_ckp_logs', prefix= 'reach'):
    saves = os.listdir(folder)
    prev_steps = 0
    for save in saves:
        save : str
        candidate, ext = os.path.splitext(save)
        steps = int(candidate.split('_')[-2])
        if steps > prev_steps:
            chosen_file = candidate
            chosen_ext = ext
            prev_steps = steps

    return f'{folder}/{chosen_file}'


def parse_args():
    parser = argparse.ArgumentParser(description='EnvWithCamera')
    parser.add_argument('--cnn_dims', nargs=2, type=int, default=[12,24])
    parser.add_argument('--parallel_envs', type=int, default= 16)
    parser.add_argument('--name', type= str, default='default')
    parser.add_argument('--load_at_steps', type= int, default= 0)
    parser.add_argument('--n_steps', type= int, default= 1024, help='Max steps per epoch')
    parser.add_argument('-d', '--dynamic', action='store_true', help= 'If flag is present the environment will generate a moving obstacle')
    parser.add_argument('-fe', '--follow_effector', action='store_true', help='If flag is present the camera will track the effector')

    return parser.parse_args()

def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
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
            # experiments=params['experiments'],
            )
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__=='__main__':
    args = parse_args()

    params = {
    'is_render': False, 
    'is_good_view': False,
    'is_train' : True,
    'show_boundary' : False,
    'add_moving_obstacle' : args.dynamic,
    'moving_obstacle_speed' : 0.25,
    'moving_init_direction' : -1,
    'moving_init_axis' : 0,
    'workspace' : [-0.4, 0.4, 0.3, 0.7, 0.2, 0.4],
    'max_steps_one_episode' : 1024,
    'num_obstacles' : 12,
    'prob_obstacles' : 0.5,
    'obstacle_box_size' : [0.04,0.04,0.002],
    'obstacle_sphere_radius' : 0.04,
    'camera_args' : {
        'placement' : 'duo',
        'type' : 'rgb',
        'prev_pos' : 0,
        'visualize' : True,
        'follow_effector' : args.follow_effector
    },
    'debug' : False,
    #'experiments' : experiments,
    }
    

    # Separate evaluation env
    eval_env = Env(
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
        # camera_args={
        #     'placement' : 'ring',
        #     'type' : 'rgbd',
        #     'position' : 0,
        #     'visualize' : False,
        # },
        debug=params['debug'],
        #experiments=params['experiments'],
        )
    eval_env = Monitor(eval_env)
    # load env
    env = SubprocVecEnv([make_env(i) for i in range(args.parallel_envs)])
    # Stops training when the model reaches the maximum number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e8, verbose=1)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./models/best_reach_ppo/{params["camera_args"]["type"]}/{args.name}',
                       log_path=f'./models/best_reach_ppo/{args.name}', eval_freq=10000,
                       deterministic=True, render=False)
    
    # Save a checkpoint every ? steps
    checkpoint_callback = CheckpointCallback(save_freq=2e6/args.parallel_envs, save_path=f'./models/reach_ppo_ckp_logs/{params["camera_args"]["type"]}/{args.name}',
                                        name_prefix='reach')
    # Create the callback list
    callback = CallbackList([checkpoint_callback, callback_max_episodes, eval_callback])


    # assert next(model.get_parameters()).is_cuda, 'Model not in GPU'
    # model.load(get_last_save())
    # model = PPO.load('./models/reach_ppo_ckp_logs/reach_1024000_steps', env=env)


    if args.load_at_steps == 0:
        policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=128, cnn_dims= args.cnn_dims),
        )
        model = PPO("MultiInputPolicy", env, n_steps= args.n_steps, gamma=0.9918651237, policy_kwargs= policy_kwargs, batch_size=256, verbose=1, tensorboard_log=f'./models/reach_ppo_tf_logs/{params["camera_args"]["type"]}/{args.name}')
        print('New model generated.')
    else:
        # model = PPO.load(f'./models/reach_ppo_ckp_logs/rgb/v5/reach_{args.load_at_steps}_steps.zip', n_steps= args.n_steps, gamma= 0.99, env=env, tensorboard_log=f'./models/reach_ppo_tf_logs/{params["camera_args"]["type"]}/{args.name}')
        model = PPO.load(f'../../from_server/rgb/v5/reach_24000000_steps.zip', n_steps= args.n_steps, gamma= 0.987, env=env, tensorboard_log=f'./models/reach_ppo_tf_logs/{params["camera_args"]["type"]}/{args.name}')
        # model = PPO.load(f'from_server/rgb/v5/reach_24000000_steps.zip', n_steps= args.n_steps, gamma= 0.99, env=env, tensorboard_log=f'./models/reach_ppo_tf_logs/{params["camera_args"]["type"]}/{args.name}')
        print('Model loaded')
#%%
    model.learn(
        total_timesteps=40e6,
        n_eval_episodes=64,
        callback=callback,
        )
    model.save('./models/reach_ppo')

# %%
