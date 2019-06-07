#from multiprocessing import Queue
import numpy as np
from multiprocessing import Manager, Pool, Lock, cpu_count
from psapso.helper import initiate_swarm, evaluate_swarm, Bunch, make_global
from test_functions import TestFunctions

def parallel_sapso(parameters):
    '''Parallel Semi Autonomous Particle Swarm Optmizer'''

    # Import parameters to the local namespace. Syntax: p.field_name
    p = Bunch(parameters)
    limit = p.stagnation_limit
    mi = p.migration_interval
    n_dims = p.n_dimensions
    epsilon_2 = 1e-5    
    min_, max_ = getattr(TestFunctions(), p.f_name +'_space')
    diagonal_length = np.linalg.norm([max_ - min_ for _ in range(n_dims)])
    z_component = (p.max_inertia - p.min_inertia) / p.m
    v_max = abs(max_ - min_) / 2               
    function = getattr(TestFunctions(), p.f_name)

    # Initiate a group of sarms:
    group_of_swarms = [initiate_swarm(p.n, n_dims, max_, min_, function, v_max, p.c1, p.c2, p.epsilon, epsilon_2, p.c_max, p.d_low, p.d_high) for _ in range(p.n_swarms)]
    
    # Initiate the pool with global parameters:
    manager= Manager()
    acess_info_lock = Lock()
    best_fitness = manager.Value(np.float64, float('inf'))
    best_position = manager.Value(np.ndarray, np.array([float('inf')]*n_dims))
    all_bests = manager.list()
    # processes=p.n_swarms,
    particle_pool = Pool(initializer=make_global, initargs=(p.n, p.m, n_dims, diagonal_length, p.max_inertia, z_component, p.minimum_improvement, best_fitness, best_position, all_bests, acess_info_lock, limit, mi))
    
    # Map swarms to individual processes and evaluate them:
    results = particle_pool.map(evaluate_swarm, group_of_swarms)
    particle_pool.close()
    particle_pool.join()

    return all_bests