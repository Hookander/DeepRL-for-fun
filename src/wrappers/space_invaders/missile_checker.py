

"""
This class is responsible for checking if a missile is coming towards the player.
Basicly there is just a memory from the last state and we check whether the missile 
went down or not. If it did, then it's coming towards the player.
"""

class MissileChecker:
    def __init__(self):
        self.last_state = None

    def check_missile(self, state):
        if self.last_state is not None:
            #The missile is coming towards the player
            if self.last_state[187][90:93].sum() == 0 and state[187][90:93].sum() > 0:
                return True
        self.last_state = state
        return False