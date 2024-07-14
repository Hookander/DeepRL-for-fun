from src.wrappers.repeat_wrapper import RepeatActionV0
from src.wrappers.space_invaders.space_invaders_wrapper import SpaceInvadersWrapper
from src.wrappers.test_wrapper import TestWrapper
from src.wrappers.breakout.breakout_wrapper import BreakoutWrapper
from src.wrappers.compatibilities import compatible_wrappers

wrapper_name_to_WrapperClass = {
    'RepeatActionV0': RepeatActionV0,
    'SpaceInvadersWrapper': SpaceInvadersWrapper,
    'TestWrapper': TestWrapper,
    "BreakoutWrapper": BreakoutWrapper}