import numpy as np


class MED:
    """
    Mobile Edge Device:
    z: column vector of predictions of actions of all MEDs (including self)
    x: action of this MED
    """

    def __init__(self, users, max_local_resource, id):
        # initializing to 0 may lead to updating failure (0->0->0...)
        self.z = np.array([np.random.uniform(0,max_local_resource[i]) for i in range(users)])
        self.x = np.random.uniform(0,max_local_resource[id])
        self.received_z = np.zeros(shape=(users,users)) # received_z[j]: the z_j received from MED j
        self.received_z[id] = self.z

        # attributes from previous iteration
        self.z_prev = np.zeros(users)
        self.x_prev = 0

def init_devices(users, max_local_resource):
    devices = []
    for i in range(users):
        devices.append(MED(users, max_local_resource, i))
    return devices

