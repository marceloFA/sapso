import numpy as np

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


def calculate_diversity(swarm, n, L):
    """ Calculates dieversity of the swarm
      swarm = gorup of particles (n x n_dims)
      L = diagonal length of the search space (scalar)
      n = number of particles (scalar)
    """
    mean = np.mean(swarm, axis=0)
    minus_mean = np.array([particle - mean for particle in swarm]) ** 2
    factor = np.sum(np.sqrt(np.sum(minus_mean, axis=1)))
    diversity = 1. / (n * L) * factor
    return diversity


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
    return dir_, importance


def get_gradient_parallel(particle):
    ''' Calculates the gradient of an individual particle
        For parallel calculation
    '''
    step = 1e-5
    steps = np.array([np.linspace(x, x+step, global_grad_params['n_dims']) for x in particle]).transpose()
    grid = list(map(global_grad_params['f'],steps))
    gradient = np.gradient(grid)
    
    #validate it:
    v_max = global_grad_params['v_max']
    gradient[gradient > v_max] = v_max
    gradient[gradient < -v_max] = -v_max
    return gradient

def get_gradient_seq(particle, function, n_dims, v_max):
    ''' Calculates the gradient of an individual particle
        For sequencial execution
    '''
    step = 1e-5
    steps = np.array([np.linspace(x, x+step, n_dims) for x in particle]).transpose()
    grid = list(map(function,steps))
    gradient = np.gradient(grid)

    #validate it:
    gradient[gradient > v_max] = v_max
    gradient[gradient < -v_max] = -v_max
    return gradient


def calculate_gradient(swarm, function,v_max, n_dims, pool, chunksize, parallel_execution):
    '''Calculates Gradient information'''
    if parallel_execution:
        gradient = np.array(pool.map(get_gradient_parallel, swarm, chunksize=chunksize))
    else:
        gradient = [get_gradient_seq(particle, function, n_dims, v_max) for particle in swarm]

    return gradient
        

def calculate_velocity(velocity, particle, importance, gradient, n_dims, inertia, c1, c2, best_position, v_max, dir_):
    ''' Calculates swarm velocity
    c1, c2 = social and cognitive components (global var)
    dir_ = direction of particle's movement (global var)

    n_dims (global)'''
    size = n_dims * 2
    phi = np.random.standard_normal(size=size)
    phi_1 = phi[:size // 2]
    phi_2 = phi[size // 2:]
    component_1 = importance * c1 * phi_1 * (best_position - particle)
    component_2 = (importance - 1) * c2 * phi_2 * gradient
    velocity = (inertia * velocity) + dir_ * (component_1 + component_2)
    
    # Validate it:
    velocity[velocity > v_max] = v_max
    velocity[velocity < -v_max] = -v_max
    
    return velocity


def update_position(particle, velocity, importance, counter, min_, max_):
    ''' Updates all particle's positions based on their velocity
        Also validate it checking if search space boundaries were exeeded.
    '''
    particle += velocity
    # Validate it:
    for k in range(len(particle)):
        if particle[k] < min_:
            particle[k] = min_
            importance = 1
            c = 0

        elif particle[k] > max_:
            particle[k] = max_
            importance = 1
            c = 0


def update_best_global(particle, fitness, best_fitness, best_position):
    '''After an iteration the best fitness found must be updated'''
    if fitness < best_fitness:
        best_fitness = np.copy(fitness)
        best_position = np.copy(particle)
    return best_fitness, best_position


def stop_condition(stop_counter,best_fitness,last_best_fitness,stop):    
    if np.all((best_fitness - last_best_fitness) < stop):
        stop_counter += 1
    return stop_counter

class Bunch(object):
    '''Saves parameters to the local namespace where
    it's called as individual variables'''
    def __init__(self, object):
        self.__dict__.update(object)

def make_global(grad_params):
    ''' Used to pass global vars in a dict format
        to a work pool'''
    global global_grad_params
    global_grad_params = grad_params