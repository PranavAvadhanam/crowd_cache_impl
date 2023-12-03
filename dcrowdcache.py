"""(Serial) Implementation of Algorithm 2 (& 1), Decentralized CrowdCache, in "CrowdCache: A Decentralized Game-Theoretic
Framework for Mobile Edge Content Sharing" by Nguyen et al."""

import numpy as np
import ComGraphGen
import med
import time
import random
import multiprocessing as mp

# set training parameters
users = 512
iters = 50
step_size = 1  # adjust depending on users and iters (for 500 users, ~2000 iters: set to 20)
heavy_ball_weight = 0.8
price_weight = 0.5  # 'gamma'
max_reward = 1
max_local_resource = [random.choice((16, 32, 48, 64)) for i in range(users)]  # C_i, in GB
quad_cost_coeff = [random.uniform(.01, .1) for i in range(users)]  # $/hr.
lin_cost_coeff = [random.uniform(.05, .15) for i in range(users)]  # $/hr.


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


def gradient_objective(i, weighted_z, devices, quad_cost_coeff, lin_cost_coeff, price_weight, max_reward):
    """ith partial derivative of objective function J for device i"""
    deriv = 2 * (quad_cost_coeff[i] + price_weight) * weighted_z[i] + lin_cost_coeff[i] - max_reward
    for j in range(len(devices)):
        if j != i:
            deriv = deriv + price_weight * weighted_z[j]
    return deriv


def iterate_user(user_devices, metropolis_weights, i):
    """
    :return: updated MED object
    """

    z_i_excl_self = np.append(user_devices[i].z[0:i], user_devices[i].z[i + 1:users])
    others_estimates = np.zeros(users - 1)
    for j in range(users):
        z_excl_self = np.append(metropolis_weights[i][j] * user_devices[j].z[0:i],
                                metropolis_weights[i][j] * user_devices[j].z[
                                                           i + 1:users])  # current estimate of all other x_j's
        others_estimates = others_estimates + z_excl_self
    # heavy-ball momentum
    others_estimates = others_estimates + heavy_ball_weight * (z_i_excl_self - user_devices[i].z_excl_self_prev)
    user_devices[i].z_excl_self_prev = z_i_excl_self

    # update action x_i
    new_action = 0
    for j in range(users):
        new_action = new_action + metropolis_weights[i][j] * user_devices[j].z[i]
    weighted_z = np.zeros(users)
    for j in range(users):
        weighted_z = metropolis_weights[i][j] * user_devices[j].z
    new_action = new_action - step_size * gradient_objective(i, weighted_z, user_devices, quad_cost_coeff,
                                                             lin_cost_coeff,
                                                             price_weight, max_reward)
    # heavy-ball momentum
    new_action = new_action + heavy_ball_weight * (user_devices[i].x - user_devices[i].x_prev)
    # ensure new_action is in range [0, C_i]
    if new_action < 0:
        new_action = 0
    if new_action > max_local_resource[i]:
        new_action = max_local_resource[i]
    user_devices[i].x_prev = user_devices[i].x
    user_devices[i].x = new_action

    # update z_i with others_estimates
    other_count = 0
    for j in range(users):
        if i == j:
            user_devices[i].z[j] = user_devices[i].x
        else:
            user_devices[i].z[j] = others_estimates[other_count]
            other_count = other_count + 1

    return user_devices[i]


if __name__ == '__main__':
# def run():
    start = time.time()

    devices = med.init_devices(users)
    # generate communication graphs for each iteration
    metro_weights = np.zeros((users, users))

    for it in range(iters):
        # generate communication graph for the iteration
        graph = ComGraphGen.generate_graph(user_count=users)

        # calculate metropolis weights for iteration
        for i in range(users):
            for j in range(users):
                metro_weights[i][j] = metropolis_weight(i, j, graph, users=users)

        # create processes
        pool = mp.Pool()  # need to restrict # processes (to ~ # cores) to avoid overhead of OS provisioning too many processes
        devices = pool.starmap(iterate_user, [(devices, metro_weights, i) for i in range(users)])

        # for i in range(users):
        #     iterate_user(devices, metro_weights, i)

    for d in devices:
        print(d.x)
    print(time.time() - start)


# 113 sec.: 500 users, 10 iterations
# 917 sec.: 500 users, 50 iterations