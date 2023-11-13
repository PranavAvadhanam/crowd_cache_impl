"""(Serial) Implementation of Algorithm 2 (& 1), Decentralized CrowdCache, in "CrowdCache: A Decentralized Game-Theoretic
Framework for Mobile Edge Content Sharing" by Nguyen et al."""

import numpy as np
import ComGraphGen
import med
import random


def metropolis_weight(i, j, adj_matrix, users):
    if (i != j) and adj_matrix[i][j] == 1:
        return 1 / (1 + max(np.count_nonzero(adj_matrix[i]), np.count_nonzero(adj_matrix[j])))
    if adj_matrix[i][j] == 0:
        return 0
    if i == j:
        out = 1
        for j in range(users):
            if j != i and adj_matrix[i][j] == 1:
                out = out - metropolis_weight(i, j, adj_matrix, users)
        return out


def init_devices(users):
    devices = []
    for i in range(users):
        devices.append(med.MED(users))
    return devices


def gradient_objective(i, weighted_z, devices, quad_cost_coeff, lin_cost_coeff, price_weight, max_reward):
    """ith partial derivative of objective function J for device i"""
    deriv = 2 * (quad_cost_coeff[i] + price_weight) * weighted_z[i] + lin_cost_coeff[i] - max_reward
    for j in range(len(devices)):
        if j != i:
            deriv = deriv + price_weight * weighted_z[j]
    return deriv


# set training parameters
users = 12
iters = 5
step_size = 1 # adjust depending on users and iters (for 500 users, ~2000 iters: set to 20)
heavy_ball_weight = 0.8
price_weight = 0.5  # 'gamma'
max_reward = 1
max_local_resource = [random.choice((16, 32, 48, 64)) for i in range(users)]  # C_i, in GB
quad_cost_coeff = [random.uniform(.01, .1) for i in range(users)]  # $/hr.
lin_cost_coeff = [random.uniform(.05, .15) for i in range(users)]  # $/hr.

devices = init_devices(users)

# generate communication graphs for each iteration
ComGraphGen.main(users=users, iters=iters)

for it in range(iters):
    # calculate metropolis weights for iteration
    metro_weights = np.zeros((users, users))
    for i in range(users):
        for j in range(users):
            metro_weights[i][j] = metropolis_weight(i, j, np.loadtxt('output/' + str(it + 1) + '/com_graph.csv',
                                                                     delimiter=','), users=users)

    # calculate others_estimates: device i's estimates of all the other devices' (excluding i) actions x
    for i in range(users):
        z_i_excl_self = np.append(devices[i].z[0:i], devices[i].z[i+1:users])
        others_estimates = np.zeros(users - 1)
        for j in range(users):
            z_excl_self = np.append(metro_weights[i][j] * devices[j].z[0:i],
                                    metro_weights[i][j] * devices[j].z[
                                                          i + 1:users])  # current estimate of all other x_j's
            others_estimates = others_estimates + z_excl_self
        # heavy-ball momentum
        others_estimates = others_estimates + heavy_ball_weight * (z_i_excl_self - devices[i].z_excl_self_prev)
        devices[i].z_excl_self_prev = z_i_excl_self

        # update action x_i
        new_action = 0
        for j in range(users):
            new_action = new_action + metro_weights[i][j] * devices[j].z[i]
        weighted_z = np.zeros(users)
        for j in range(users):
            weighted_z = metro_weights[i][j]*devices[j].z
        new_action = new_action - step_size * gradient_objective(i, weighted_z, devices, quad_cost_coeff, lin_cost_coeff,
                                                                 price_weight, max_reward)
        # heavy-ball momentum
        new_action = new_action + heavy_ball_weight * (devices[i].x - devices[i].x_prev)
        # ensure new_action is in range [0, C_i]
        if new_action < 0:
            new_action = 0
        if new_action > max_local_resource[i]:
            new_action = max_local_resource[i]
        devices[i].x_prev = devices[i].x
        devices[i].x = new_action

        # update z_i with others_estimates
        other_count = 0
        for j in range(users):
            if i == j:
                devices[i].z[j] = devices[i].x
            else:
                devices[i].z[j] = others_estimates[other_count]
                other_count = other_count + 1

print(devices[0].x, devices[4].x)