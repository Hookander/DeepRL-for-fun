from src.wrappers.repeat_wrapper import RepeatActionV0
from src.wrappers.space_invaders.detect_death import DetectDeathV0
from src.wrappers.compatibilities import compatible_wrappers

wrapper_name_to_WrapperClass = {
    'RepeatActionV0': RepeatActionV0,
    'DetectDeathV0': DetectDeathV0}