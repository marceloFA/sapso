#from multiprocessing import Queue
import numpy as np
from multiprocessing import Manager, Pool, Lock, cpu_count
from auxiliar_new_version import initiate_swarm, evaluate_swarm, Bunch, make_global
from test_functions import TestFunctions

def pso(parameters):
    '''psapso algorithm'''
    # parameters down here:
    p = Bunch(parameters)
    n_swarms = p.n_swarms
    limit = p.stagnation_limit
    mi = p.migration_interval
    n_dims = p.n_dimensions
    c1 = p.c1
    c2 = p.c2
    c_max = p.c_max
    epsilon_1 = p.epsilon
    d_low = p.d_low
    d_high = p.d_high
    f_name = p.f_name
    epsilon_2 = 1e-5    
    min_, max_ = getattr(TestFunctions(), f_name +'_space')
    diagonal_length = np.linalg.norm([max_ - min_ for _ in range(n_dims)])
    z_component = (p.max_inertia - p.min_inertia) / p.m
    v_max = abs(max_ - min_) / 2               
    function = getattr(TestFunctions(), f_name)

    
    # Initiates a group of sarms:
    group_of_swarms = [initiate_swarm(p.n, n_dims, max_, min_, function, v_max, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high) for _ in range(n_swarms)]
    
    # Initiate the pool with global parameters:
    manager= Manager()
    acess_info_lock = Lock()
    best_fitness = manager.Value(np.float64, 1.)
    best_position = manager.Value(np.ndarray, np.array([1.]*n_dims))
    all_bests = manager.list()
    particle_pool = Pool(processes=n_swarms, initializer=make_global, initargs=(p.n, p.m, n_dims, diagonal_length, p.max_inertia, z_component, p.minimum_improvement, best_fitness, best_position, all_bests, acess_info_lock, limit, mi))
    
    # Map swarms to individual processes:
    results = particle_pool.map(evaluate_swarm, group_of_swarms, chunksize=int(n_swarms/cpu_count()))
    particle_pool.close()
    particle_pool.join()
    # filter results before returning then:
    best_fitnesses = [min([particle.fitness for particle in swarm]) for swarm in results]
    print(shared_all_bests)
    return min(best_fitnesses)