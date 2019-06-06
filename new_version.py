#from multiprocessing import Queue
import numpy as np
from auxiliar_new_version import initiate_swarm, evaluate_swarm, prepare_eval_pool, Bunch
from test_functions import TestFunctions

def pso(parameters):
    '''psapso algorithm'''
    # parameters down here:
    p = Bunch(parameters)
    #global z, L
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

    # Optimizer components initialization:
    group_of_swarms = [initiate_swarm(p.n, n_dims, max_, min_, function, v_max, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high) for _ in range(n_swarms)]
    
    # Define variables that will be passed to the evaluation pool and generate the pool itself:
    particle_pool = prepare_eval_pool(p.n, p.m, n_dims, diagonal_length, p.max_inertia, z_component, p.minimum_improvement, limit, mi)
    
    # Map swarms to individual processes:
    results = particle_pool.map(evaluate_swarm,group_of_swarms)
    # filter results before returning then:
    best_fitnesses = [min([particle.fitness for particle in swarm]) for swarm in results]
    results = {
        'best_fitnesses': best_fitnesses,
        'global_best_fitness': min(best_fitnesses),
    }

    return results