from src.wrappers import *

"""
Here we make sure that the wrappers asked for are compatible with the environment.
If they are not, we won't apply them in the trainer init method.
"""

compatible_wrappers = {
    'ALE/SpaceInvaders-v5': [RepeatActionV0, SpaceInvadersWrapper, TestWrapper],
    'CartPole-v1': [RepeatActionV0],
    'CarRacing-v2': [RepeatActionV0],}