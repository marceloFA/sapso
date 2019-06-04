#from multiprocessing import Queue
import numpy as np
from test_functions import TestFunctions
from auxiliar_new_version import * # check module to its content

def pso(parameters):
    '''psapso algorithm'''
    # parameters down here:
    p = Bunch(parameters)
    n = p.n
    n_iters = p.m
    stop = p.stop
    n_dims = p.n_dimensions
    min_inertia = p.min_inertia
    max_inertia = p.max_inertia
    c1 = p.c1
    c2 = p.c2
    c_max = p.c_max
    epsilon_1 = p.epsilon
    d_low = p.d_low
    d_high = p.d_high
    f_name = p.f_name
    diversity = 0. 
    importance = 1
    epsilon_2 = 1e-5    
    min_, max_ = getattr(TestFunctions(), f_name +'_space')
    L = np.linalg.norm([max_ - min_ for _ in range(n_dims)])
    z = (max_inertia - min_inertia) / n_iters   
    v_max = abs(max_ - min_) / 2               
    function = getattr(TestFunctions(), f_name)
    stagnation = 0
    limit = 500 # n_iters with no significant improvement on best position

    # Optimizer components initialization:
    swarm = initiate_swarm(n, n_dims, max_, min_, function, v_max, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high)
    fitness_list = np.array([particle.fitness for particle in swarm])
    best_fitness = np.amin(fitness_list)
    best_position = fitness_list[np.where(fitness_list == best_fitness)][0]

    # Optmization Loop:
    for i in range(n_iters):
        # Global information:
        diversity = calculate_diversity(swarm, n, L)
        inertia = (max_inertia - i) * z
        last_best_fitness = best_fitness.copy()

        for particle in swarm:
            # Move particles:
            particle.adjust_direction(diversity)
            particle.update_velocity(inertia, best_position)
            particle.update_position()
            particle.update_fitness()
            # update global best:
            if particle.fitness < best_fitness:
                best_fitness = particle.fitness
                best_position = particle.position
            #sapso information:
            particle.update_importance(best_position)

        # Stop condition:
        stagnation += 1 if stop_condition(best_fitness, last_best_fitness, stop) else 0
        if stagnation >= limit: 
            break

    return best_fitness, best_position, i