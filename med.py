import numpy as np


class MED:
    """
    Mobile Edge Device:
    z: column vector of predictions of actions of all MEDs (including self)
    x: action of this MED
    """

    def __init__(self, users):
        self.z = np.zeros(users)
        self.x = 0
        # attributes from previous iteration
        self.z_excl_self_prev = np.zeros(users - 1)  # previous iter's estimates, excluding self
        self.x_prev = 0
