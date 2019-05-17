# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Pool, cpu_count
from test_functions import TestFunctions

def get_velocity(v, p, i, g, inertia, dir_, best_position):
    ''' Calculates swarm velocity
    c1, c2 = social and cognitive components (global var)
    
    p = particle
    v = velocity
    i = importance
    g = gradient
    '''
    size = global_params['n_dims'] * 2
    phi = np.random.standard_normal(size=size)
    phi_1 = phi[:size // 2]
    phi_2 = phi[size // 2:]
    component_1 = i * global_params['c1'] * phi_1 * (best_position - p)
    component_2 = (i - 1) * global_params['c2'] * phi_2 * g
    velocity = (inertia * v) + dir_ * (component_1+component_2) 
    return velocity


def get_gradient(particle):
    ''' Calculates the gradient of an individual particle
        For parallel calculation
    '''
    step = 1e-5
    steps = np.array([np.linspace(x, x+step, global_params['n_dims']) for x in particle]).transpose()
    grid = list(map(global_params['f'],steps))
    gradient = np.gradient(grid)
    
    #validate it:
    v_max = global_params['v_max']
    gradient[gradient > v_max] = v_max
    gradient[gradient < -v_max] = -v_max
    return gradient

                 
def calculate_velocity(velocity, swarm, importance, gradient, inertia, best_position, v_max, dir_, n, pool):
    '''Calculates swarm velocities'''
    # Expensive initiation, do not know how to make it computationally cheaper!
    velocity = pool.starmap(get_velocity, zip(velocity, swarm, importance, gradient, [inertia]*n, [dir_]*n, [best_position]*n))
    velocity = np.array(velocity)
    #Validate it:
    velocity[velocity > v_max] = v_max
    velocity[velocity < -v_max] = -v_max
    
    return velocity

def calculate_gradient(swarm, pool):
    '''Calculates Gradient information'''
    gradient = np.array(pool.map(get_gradient,swarm))
    return gradient


def update_swarm(swarm,velocity, importance, counter,n, n_dims, min_, max_):
    ''' Updates all particle's positions based on their velocity
        Also validate it checking if search space boundaries were exeeded.
    '''
    for i in range(n):
        swarm[i] += velocity[i]
        for j in range(n_dims):
            if swarm[i][j] < min_:
                swarm[i][j] = min_
                importance[i] = 1
                counter[i] = 0

            elif swarm[i][j] > max_:
                swarm[i][j] = max_
                importance[i] = 1
                counter[i] = 0

def update_fitness(swarm, function, pool):
    '''Get new cost for all particles in the swarm'''
    return np.array(pool.map(function,swarm))

def update_best_found(swarm, fitness, best_fitness, best_position, n):
    '''After an iteration the best fitness found must be updated'''
    for i in range(n):
        if fitness[i] < best_fitness:
            best_fitness = np.copy(fitness[i])
            best_position = np.copy(swarm[i])
    return best_fitness, best_position


def update_importance(importance, swarm, fitness, last_fitness, counter, best_position, n, c_max, epsilon, epsilon_2):
    '''After an iteration importance for each particle must be updated'''
    for k in range(n):
        # Check to see if we are improving fitness through iterations:
        if importance[k] == 0:
            if abs(fitness[k] - last_fitness[k]) <= epsilon:
                counter[k] += 1
                # If sapso can't improve fitness within c_max iterations:
                # then importance is 1 (particle will go onto the best global
                # instead of gradient information)
                if counter[k] == c_max:
                    importance[k] = 1
                    counter[k] = 0

            else:
                counter[k] = 0

        elif importance[k] == 1:
            if np.sqrt(np.sum(np.power((swarm[k] - best_position), 2))) < epsilon_2:
                importance[k] = 0
                counter[k] = 0


def calculate_dir_and_importance(importance, diversity, d_low, d_high, dir_, n):
    ''' Calculates direction and importance
        n = number of particles (global)
        dir_ = direction of particle's movement (global var)
    '''
    if (dir_ > 0 and diversity < d_low):  # must repulse
        dir_ = -1
        importance = np.ones(n)
    elif (dir_ < 0 and diversity > d_high):  # must attract
        dir_ = 1
        importance = np.zeros(n)
    return importance


def calculate_diversity(swarm, n, L):
    """ Calculates dieversity of the swarm"""
    mean = np.mean(swarm, axis=0)
    minus_mean = np.array([particle - mean for particle in swarm]) ** 2
    factor = np.sum(np.sqrt(np.sum(minus_mean, axis=1)))
    diversity = 1. / (n * L) * factor
    return diversity


def stop_condition(stop_counter,best_fitness,last_best_fitness,stop):    
    if np.all((best_fitness - last_best_fitness) < stop):
        stop_counter += 1
    return stop_counter


class Bunch(object):
    '''Saves parameters to the local namespace as individual variables'''

    def __init__(self, parameters):
        self.__dict__.update(parameters)

def make_params_global(params):
    ''' Used to pass global vars in a dict format
        to a work pool'''
    global global_params
    global_params = params


def psapso(parameters):
    '''The Parallel Semi Autonomus particle swarm optmizer'''s
    # Define some parameters:
    n, m, n_dims, min_inertia, max_inertia, c1, c2, c_max, epsilon, d_low, d_high, stop, f_name = parameters
    dir_ = 1
    diversity = 0.                
    epsilon_2 = 1e-5              
    min_, max_ = getattr(TestFunctions(), f_name +'_space')  
    z = (max_inertia - min_inertia) / m   
    v_max = abs(max_ - min_) / 2               
    function = getattr(TestFunctions(), f_name)
    counter = np.zeros(n)
    L = np.linalg.norm([max_ - min_ for _ in range(n_dims)])
    optmizer_counter = 0
    stop_counter = 100 # n iters without new best_global

    # Get dict of params to be passed to work pools according to task needs:
    params = {'n_dims':n_dims, 'v_max':v_max, 'f':function, 'c1':c1, 'c2':c2, 'v_max':v_max}
    
    # Instantiate Pools:
    cores = cpu_count()
    cost_work_pool = Pool(cores)
    velocity_work_pool = Pool(cores, initializer=make_params_global, initargs=(params,))
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
    gradient = calculate_gradient(swarm, grad_work_pool)
    
    # 1 Main Loop:   
    for i in range(m):
        last_fitness = np.copy(fitness)
        last_best_fitness = np.copy(best_fitness)
        inertia = (max_inertia - i) * z
        
        gradient = calculate_gradient(swarm, grad_work_pool)
        velocity = calculate_velocity(velocity, swarm, importance, gradient, inertia, best_position, v_max, dir_,n, velocity_work_pool)
        update_swarm(swarm,velocity, importance, counter,n, n_dims, min_, max_)
        fitness = update_fitness(swarm, function, cost_work_pool)
        best_fitness, best_position = update_best_found(swarm, fitness, best_fitness, best_position, n)
        
        # SAPSO information:
        update_importance(importance, swarm, fitness, last_fitness, counter, best_position, n, c_max, epsilon, epsilon_2)
        diversity = calculate_diversity(swarm, n, L)
        importance = calculate_dir_and_importance(importance, diversity, d_low, d_high, dir_, n)
        gradient = calculate_gradient(swarm, grad_work_pool)
        stop_counter = stop_condition(stop_counter,best_fitness,last_best_fitness,stop)
        # Stop criterion
        if optmizer_counter > stop_counter: break
    return best_position, best_fitness