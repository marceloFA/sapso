# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Pool, cpu_count
from test_functions import TestFunctions
from auxiliar_psapso import *

def psapso(parameters):
    '''The Parallel Semi Autonomous particle swarm optmizer'''
    # Define some parameters:
    p = Bunch(parameters)
    n = p.n
    m = p.m
    stop = p.stop
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

    dir_ = 1
    diversity = 0.                
    epsilon_2 = 1e-5              
    min_, max_ = getattr(TestFunctions(), f_name +'_space')  
    z = (max_inertia - min_inertia) / m   
    v_max = abs(max_ - min_) / 2               
    function = getattr(TestFunctions(), f_name)
    counter = np.zeros(n)
    L = np.linalg.norm([max_ - min_ for _ in range(n_dims)])
    #stagnation = 0
    #limit = 50 # n_iters with no significant improvement on best position

    # Get dict of params to be passed to work pools according to tasks needs:
    params = {'n_dims':n_dims, 'v_max':v_max, 'f':function, 'c1':c1, 'c2':c2, 'v_max':v_max}
    
    # Instantiate Pools:
    cores = cpu_count()
    chunksize = int(n/cores)
    cost_work_pool = Pool(cores)
    vel_work_pool = Pool(cores, initializer=make_params_global, initargs=(params,))
    grad_work_pool = Pool(cores, initializer=make_params_global, initargs=(params,))
    
    # Initialize components:
    velocity = np.zeros((n, n_dims))
    gradient = np.zeros((n, n_dims))
    swarm = np.array([[min_ + np.random.uniform() * (max_ - min_)
                       for i in range(n_dims)] for _ in range(n)])
    importance = np.ones(n)
    fitness = np.array(list(map(function, swarm)))
    best_fitness = np.amin(fitness)
    best_position = swarm[np.where(fitness == best_fitness)][0]    
    gradient = calculate_gradient(swarm, grad_work_pool, chunksize)
    
    # 1 Main Loop:   
    for i in range(m):
        last_fitness = np.copy(fitness)
        #last_best_fitness = np.copy(best_fitness)
        inertia = (max_inertia - i) * z
        
        velocity = calculate_velocity(velocity, swarm, importance, gradient, inertia, best_position, v_max, dir_,n, vel_work_pool, chunksize)
        update_swarm(swarm,velocity, importance, counter,n, n_dims, min_, max_)
        fitness = update_fitness(swarm, function, cost_work_pool, chunksize)
        best_fitness, best_position = update_best_found(swarm, fitness, best_fitness, best_position, n)
        
        # SAPSO information:
        update_importance(importance, swarm, fitness, last_fitness, counter, best_position, n, c_max, epsilon, epsilon_2)
        diversity = calculate_diversity(swarm, n, L)
        importance, dir_ = calculate_dir_and_importance(importance, diversity, d_low, d_high, dir_, n)
        gradient = calculate_gradient(swarm, grad_work_pool, chunksize)
        #if update_stagnation(best_fitness,last_best_fitness,stop):
        #    stagnation += 1
        
        # Stop criterion:
        #if stagnation >= limit: 
            #break
    return best_position, best_fitness