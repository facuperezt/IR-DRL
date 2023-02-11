#%%
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class CustomizableCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, cnn_out_channels = None, features_dim: int = 128):
        self.n_frames = 1
        if cnn_out_channels is None:
            cnn_out_channels = [16, 32, 32]
        assert type(cnn_out_channels) is list and len(cnn_out_channels) == 3, 'cnn_out_channels has to be a list of len 3'
        if len(observation_space.shape) == 3:
            n_input_channels = observation_space.shape[0]
            sample = observation_space.sample()[None]
        elif len(observation_space.shape) == 4:
            n_input_channels = observation_space.shape[-1]
            self.n_frames = observation_space.shape[0]
            # assert features_dim/self.n_frames == features_dim//self.n_frames, 'features_dim has to be divisible by n_frames'
            sample = observation_space.sample().transpose(0,3,1,2)
        super(CustomizableCNN, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, cnn_out_channels[0], kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(cnn_out_channels[1], cnn_out_channels[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(sample).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        self.future_biased_weighting = (((th.arange(self.n_frames) + 1)/(2*self.n_frames)) + 0.5).view(1,-1,1) # should be a line from 0.5 for the oldest frame to 1 for the newest one

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.n_frames == 1:
            out :th.Tensor = self.linear(self.cnn(observations))
        else:
            observations = observations.permute(0, 1, 4, 2, 3)
            observations = observations.reshape(-1, *observations.shape[2:])
            out :th.Tensor = self.linear(self.cnn(observations)).reshape(observations.shape[0]//self.n_frames, self.n_frames, -1)

            if out.device != self.future_biased_weighting.device:
                self.future_biased_weighting = self.future_biased_weighting.to(out.device)    

            out = (self.future_biased_weighting * out).mean(dim=1)
        
        return out

class ObstacleRadiusExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, linear_sizes = None, cnn_sizes= None, features_dim: int = None):
        if linear_sizes is None:
            linear_sizes = [int(observation_space.shape[0] * observation_space.shape[1]/2), int(observation_space.shape[0] * observation_space.shape[1]/4)]
        if features_dim is None:
            features_dim = int(observation_space.shape[0] * observation_space.shape[1] / 8)
        if cnn_sizes is None:
            cnn_sizes = [2, 1]

        super().__init__(observation_space, features_dim)

        # cnn_channel_sizes = [observation_space.shape[0], *cnn_sizes]
        # cnn_layers = []
        # for i in range(len(cnn_channel_sizes) - 1):
        #     cnn_layers.append(nn.Conv1d(cnn_channel_sizes[i], cnn_channel_sizes[i+1], 1))
        #     cnn_layers.append(nn.LeakyReLU())

        # self.cnn = nn.Sequential(*cnn_layers, nn.Flatten())

        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(th.as_tensor(observation_space.sample()).float()).shape[1]

        linear_layer_sizes = [observation_space.shape[0] * observation_space.shape[1], *linear_sizes, features_dim]

        linear_layers = [nn.Flatten()]
        for i in range(len(linear_layer_sizes)-1):
            linear_layers.append(nn.Linear(linear_layer_sizes[i], linear_layer_sizes[i+1]))
            linear_layers.append(nn.LeakyReLU())

        self.linear = nn.Sequential(*linear_layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(observations)


class CustomizableFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, args= None):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomizableFeaturesExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if 'camera' in key:
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = CustomizableCNN(subspace, **args)
                total_concat_size += extractors[key].features_dim
            elif 'obstacleradius' in key:
                extractors[key] = ObstacleRadiusExtractor(subspace, **args)
                total_concat_size += extractors[key].features_dim
            else:
                # Flatten it
                extractors[key] = nn.Flatten() #DummyBatchFlatten()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

class DropoutMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):

    def __init__(self, *args, dropout_rate = 0.25, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DropoutMlpExtractor(
            self.features_dim,
            self.device,
            self.dropout_rate,
        )

class DropoutMlpExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        device: Union[th.device, str] = "auto",
        dropout_rate: float = 0.5,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Shared network
        self.shared_net = nn.Sequential(

        ).to(device)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh(),
            
        ).to(device)
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), nn.Tanh(),
        ).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
# %%
# obs_space = spaces.Box(low=0, high= 1, shape=(10, 128, 128, 4))
# cnn = CustomizableCNN(obs_space)