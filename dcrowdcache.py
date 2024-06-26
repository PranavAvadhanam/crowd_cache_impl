"""Implementation of Algorithm 2 (& 1), Decentralized CrowdCache, in "CrowdCache: A Decentralized Game-Theoretic
Framework for Mobile Edge Content Sharing" by Nguyen et al."""

import os
import numpy as np
import json
import pandas
import med
import time
import random
import multiprocessing as mp
import matplotlib.pyplot as plt

# set training parameters
users = 7
iters = 4
num_batches = 6
step_size = .00004  # adjust depending on users and iters
heavy_ball_weight = 0.8
price_weight = 0.5  # 'gamma'
max_reward = 64*users
max_local_resource = [random.choice((16, 32, 48, 64)) for i in range(users)]  # C_i, in GB
quad_cost_coeff = [random.uniform(.01, .1) for i in range(users)]  # $/hr.
lin_cost_coeff = [random.uniform(.05, .15) for i in range(users)]  # $/hr.

diff_x = np.zeros(users)  # lists diff x_i - x_i-1 for given iteration
diff_mag = np.zeros(num_batches*iters)  # lists L_2 norm of diff_x array for every iteration


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


def gradient_objective(i, z_aggregate, devices):
    """ith partial derivative of objective function J for device i"""
    deriv = 2 * (quad_cost_coeff[i] + price_weight) * z_aggregate[i] + lin_cost_coeff[i] - max_reward
    for j in range(len(devices)):
        if j != i:
            deriv += price_weight * z_aggregate[j]
    return deriv


def local_objective(x, i, devices):
    """local_objective for a hypothetical x_i"""
    objective = quad_cost_coeff[i] * x * x + lin_cost_coeff[i] * x
    reward = max_reward
    for ind in range(len(devices)):
        if ind != i:
            reward = reward - price_weight * devices[ind].x
        else:
            reward = reward - price_weight * x
    return objective - reward * x


def is_nash_equilibrium(devices):
    """rough, discrete test - if returns false: guaranteed not Nash Eq; if returns true: Not necc. Nash eq."""
    all_equil = True
    for i in range(len(devices)):
        for x_alt in range(int(max_local_resource[i]) + 1):
            if (local_objective(devices[i].x, i, devices) > local_objective(x_alt, i, devices)):
                all_equil = False
    return all_equil


def iterate_user(devices, metro_weights, adj_list, i, device_updates):
    """
    :return: updated MED object
    """
    # Receive z_j from all neighbors
    for j in adj_list[i]:
        devices[i].received_z[j] = devices[j].z

    # update action x
    z_aggregate = np.zeros(users)
    for j in range(users):
        z_aggregate += metro_weights[i][j] * devices[i].received_z[j]
    self_aggr_estimate = 0
    for j in range(users):
        self_aggr_estimate += metro_weights[i][j] * devices[i].received_z[j][i]

    # if i == 324:
    #     # print(z_aggregate)
    #     print("SEFL AGGR", self_aggr_estimate)
    x_new = self_aggr_estimate - step_size * gradient_objective(i, z_aggregate, devices)
    # if i == 324:
    #     print('GRAD: ', gradient_objective(i, z_aggregate, devices))
    #     print("WEIGHT GRAD", step_size*gradient_objective(i, z_aggregate, devices))
    # bound/project onto correct range
    if x_new < 0:
        x_new = 0
    if x_new > max_local_resource[i]:
        x_new = max_local_resource[i]
    x_new += heavy_ball_weight * (devices[i].x - devices[i].x_prev)
    if x_new < 0:
        x_new = 0
    if x_new > max_local_resource[i]:
        x_new = max_local_resource[i]
    device_updates[i]['x_prev'] = devices[i].x
    device_updates[i]['x'] = x_new

    # update estimates z
    z_new = np.zeros(users)
    for j in range(users):
        z_new += metro_weights[i][j] * devices[i].received_z[j]
    z_new[i] = devices[i].x
    for j in range(users):
        if z_new[j] < 0:
            z_new[j] = 0
        if z_new[j] > max_local_resource[j]:
            z_new[j] = max_local_resource[j]
    z_new += heavy_ball_weight * (devices[i].z - devices[i].z_prev)
    for j in range(users):
        if z_new[j] < 0:
            z_new[j] = 0
        if z_new[j] > max_local_resource[j]:
            z_new[j] = max_local_resource[j]
    device_updates[i]['z_prev'] = devices[i].z
    device_updates[i]['z'] = z_new


if __name__ == '__main__':
    # def run():

    start = time.time()

    devices = med.init_devices(users, max_local_resource)
    # generate communication graphs for each iteration
    metro_weights = np.zeros((users, users))
    # pool = mp.Pool(
    # 5)  # need to restrict # processes (to ~ # cores) to avoid overhead of OS provisioning too many processes
    # manager = mp.Manager()

    for b in range(num_batches): # num_batches*iters iterations total
            
        # read data from a given iteration_batch
        outputFile = os.getcwd()+'/output/output_batch_' + str(b+1) + '.json'
        f = open(outputFile)
        dataJson = json.load(f)
        f.close()

        data = []

        for itr in dataJson:
            iteration = {}
            iteration['users'] = pandas.read_json(itr['users'], orient="split")
            iteration['com_graph'] = np.array(itr['com_graph'])
            data.append(iteration)

        for it in range(iters):
            # generate communication graph for the iteration
            s_iter = time.time()
            
            # graph = ComGraphGen.generate_graph(user_count=users)
            graph = data[it]['com_graph']
            adj_list = list([[] for i in range(users)])  # adj_list for sparse graph
            for i in range(users):
                for j in range(users):
                    if (graph[i][j] == 1):
                        adj_list[i].append(j)

            # calculate metropolis weights for iteration
            for i in range(users):
                for j in range(users):
                    metro_weights[i][j] = metropolis_weight(i, j, graph, users=users)
            # iterate w. multiprocessing
            # print('here')
            # device_updates = manager.list([manager.dict({'x_prev': 0, 'x': 0, 'z_prev': 0, 'z': 0}) for j in
            #                                range(users)])
            #
            # pool.starmap(iterate_user, [(devices, metro_weights, adj_list, i, device_updates) for i in range(users)])
            for j in range(users):
                device_updates = [{'x_prev': 0, 'x': 0, 'z_prev': 0, 'z': 0} for j in range(users)]
                iterate_user(devices, metro_weights, adj_list, j, device_updates)
                devices[j].x_prev = device_updates[j]['x_prev']
                devices[j].x = device_updates[j]['x']
                devices[j].z_prev = device_updates[j]['z_prev']
                devices[j].z = device_updates[j]['z']

            # for d in devices:
            #     print(d.x)

            # update diff_x, diff_mag
            for i in range(users):
                diff_x[i] = int(devices[i].x - devices[i].x_prev)
            diff_mag[iters*b + it] = np.linalg.norm(diff_x, ord=2)
            # print("DIFF_MAG", diff_mag[it])
            #
            # print("ITER Time:", it, time.time()-s_iter)

    for d in devices:
        print(d.x)
    print("TOTAL Time", time.time() - start)

    # Plot
    print(diff_mag)
    x = list(range(num_batches*iters))
    y = diff_mag
    fig, ax = plt.subplots()
    ax.set_title("Magnitude of difference vectors (x_i - x_i-1) across iterations")
    ax.set_xlabel('iterations')
    ax.set_ylabel('DIFF_MAG')
    ax.plot(x,y)
    plt.show()
    print(is_nash_equilibrium(devices))
    # pool.close()
