# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Pool, cpu_count
from test_functions import TestFunctions
from sapso.helper import *


def sapso(parameters):
    ''' The Semi Autonomus particle swarm optmizer. 
        This version implements a parallel calculation of 
        gradient information to improve its performance.
    '''
    # Define some parameters:
    p = Bunch(parameters)
    n = p.n
    m = p.m
    stop = p.minimum_improvement
    n_dims = p.n_dimensions
    min_inertia = p.min_inertia
    max_inertia = p.max_inertia
    c1 = p.c1
    c2 = p.c2
    c_max = p.c_max
    epsilon = p.epsilon
    d_low = p.d_low
    d_high = p.d_high
    f_name = p.f_name
    parallel = p.parallel_grad
    limit = p.stagnation_limit

    dir_ = 1
    diversity = 0.                
    epsilon_2 = 1e-5              
    min_, max_ = getattr(TestFunctions(), f_name +'_space')  
    z = (max_inertia - min_inertia) / m   
    v_max = abs(max_ - min_) / 2               
    function = getattr(TestFunctions(), f_name)
    counter = np.zeros(n)
    L = np.linalg.norm([max_ - min_ for _ in range(n_dims)])
    stagnation = 0
    
    # Initialize components:
    velocity = np.zeros((n, n_dims))
    gradient = np.zeros((n, n_dims))
    swarm = np.array([[min_ + np.random.uniform() * (max_ - min_)
                       for i in range(n_dims)] for _ in range(n)])
    importance = np.ones(n)
    fitness = np.array(list(map(function, swarm)))
    best_fitness = np.amin(fitness)
    best_position = swarm[np.where(fitness == best_fitness)][0]

    # Create group of data to be passed to a worker pool, in case its a parallel execution
    grad_params = {'n_dims':n_dims, 'v_max': v_max, 'f': function}
    chunksize = int(n/cpu_count())
    grad_work_pool = Pool(cpu_count(), initializer=make_global, initargs=(grad_params,))
    grad_skip_count = 0

    # Main loop:
    for i in range(m):
        last_fitness = np.copy(fitness)
        last_best_fitness = np.copy(best_fitness)
        inertia = (max_inertia - i) * z

        if not np.all(importance):
            gradient = calculate_gradient(swarm, function,v_max, n_dims, grad_work_pool, chunksize, parallel)

        for k in range(n):
            velocity[k] = calculate_velocity( velocity[k], swarm[k], importance[k], gradient[k], n_dims, inertia, c1, c2, best_position, v_max, dir_)
            update_position(swarm[k], velocity[k], importance[k], counter[k], min_, max_)
            fitness[k] = function(swarm[k])
            best_fitness, best_position = update_best_global(swarm[k], fitness[k], best_fitness, best_position)

        update_importance(importance, swarm, fitness, last_fitness, counter, best_position, n, c_max, epsilon, epsilon_2)
        diversity = calculate_diversity(swarm, n, L)
        dir_, importance = calculate_dir_and_importance(importance, diversity, d_low, d_high, dir_, n)
        
        # Stop criterion:
        stagnation += 1 if stop_condition(best_fitness,last_best_fitness,stop) else 0
        if stagnation >= limit: 
            break

        #print('{},{},{}'.format(diversity,dir_,best_fitness))

    grad_work_pool.close()
    grad_work_pool.terminate()
    return best_position, best_fitness