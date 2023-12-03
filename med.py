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
        self.z_excl_self_prev = np.zeros(users - 1) # previous iter's estimates, excluding self
        self.x_prev = 0

def init_devices(users):
    devices = []
    for i in range(users):
        devices.append(MED(users))
    return devices

#
# class MED:
#     """
#     Mobile Edge Device:
#     z: column vector of predictions of actions of all MEDs (including self)
#     x: action of this MED
#     """
#     #
#     # def __init__(self, users):
#     #     z = np.zeros(users)
#     #     cdef double [:] self_z = z
#     #     self.z = self_z
#     #     cdef double self_x = 0
#     #     self.x = self_x
#     #     # attributes from previous iteration
#     #     z_excl_self_prev = np.zeros(users - 1)  # previous iter's estimates, excluding self
#     #     cdef double [:] self_z_excl_self_prev = z_excl_self_prev
#     #     self.z_excl_self_prev = self_z_excl_self_prev
#     #     cdef double self_x_prev = 0
#     #     self.x_prev = self_x_prev

