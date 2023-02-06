from .dropout_policy import DropoutMultiInputActorCriticPolicy

class PolicyRegistry:
    _policy_classes = {}

    @classmethod
    def get(cls, policy_type:str):
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
