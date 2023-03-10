from .dropout_policy import DropoutMultiInputActorCriticPolicy, CustomizableFeaturesExtractor
from .CustomPPO import CustomPPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

class PolicyRegistry:
    _policy_classes = {}

    @classmethod
    def get(cls, policy_type:str) -> MultiInputActorCriticPolicy:
        try:
            return cls._policy_classes[policy_type]
        except KeyError:
            raise ValueError(f"unknown policy type : {policy_type}")

    @classmethod
    def register(cls, policy_type:str):
        def inner_wrapper(wrapped_class):
            cls._policy_classes[policy_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

PolicyRegistry.register('dropout')(DropoutMultiInputActorCriticPolicy)
PolicyRegistry.register('CustomizableFeaturesExtractor')(CustomizableFeaturesExtractor)
PolicyRegistry.register('TeacherPPO')(CustomPPO)
