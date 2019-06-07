import numpy as np
from multiprocessing import Manager, Pool, Lock, current_process

class Particle():
    ''' This class defines a particle. Particles are used to form a swarm'''

    def __init__(self, position, velocity, function, importance, direction, stagnation, v_max, n_dims, min_, max_, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high):
        ''' Initiate a particle '''
        self.position = position
        self.velocity = velocity
        self.objective_function = function
        self.fitness = self.objective_function(self.position)
        self.last_fitness = np.copy(self.fitness)
        self.gradient = np.array([0.]*n_dims)
        self.importance = importance
        self.stagnation = stagnation

        # Globally defined, locally stored
        self.direction = direction
        self.v_max = v_max
        self.n_dims = n_dims
        self.min_ = min_
        self.max_ = max_
        self.c1 = c1
        self.c2 = c2
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.c_max = c_max
        self.d_low = d_low
        self.d_high = d_high


    def update_velocity(self, inertia, best_position):
        ''' Take some parameters and calculates a particle's velocity '''
        size = self.n_dims * 2
        phi = np.random.standard_normal(size=size)
        phi_1 = phi[:size // 2]
        phi_2 = phi[size // 2:]
        component_1 = self.importance * self.c1 * phi_1 * (best_position - self.position) # problema aqui
        component_2 = (self.importance - 1) * self.c2 * phi_2 * self.gradient
        velocity = np.array((inertia * self.velocity) + self.direction * (component_1 + component_2))
    
        # Validate it:
        v_max = self.v_max
        velocity[velocity > v_max] = v_max
        velocity[velocity < -v_max] = -v_max
        # Finish updating velocity of a particle:
        self.velocity = velocity


    def update_gradient(self):
        ''' Calculate gradient of a particle '''
        # No grad neeeded:
        if self.importance == 1:
            return

        step = 1e-5
        steps = np.array([np.linspace(x, x+step, self.n_dims) for x in self.position]).transpose()
        grid = list(map(self.objective_function,steps))
        gradient = np.gradient(grid)

        #validate it:
        v_max = self.v_max
        gradient[gradient > v_max] = v_max
        gradient[gradient < -v_max] = -v_max
        
        # Finishes updating particle gradient:
        self.gradient = gradient
        

    def update_position(self):
        '''Update a particle's position '''
        self.position += self.velocity

        # Validate it:
        for k in range(self.n_dims):
            if self.position[k] < self.min_:
                self.position[k] = self.min_
                self.importance = 1
                self.stagnation = 0

            elif self.position[k] > self.max_:
                self.position[k] = self.max_
                self.importance = 1
                self.stagnation = 0
        

    def update_fitness(self):
        self.last_fitness = self.fitness
        self.fitness = self.objective_function(self.position)


    def update_importance(self, best_position):
        ''' Update the importance factor'''
        # Check to see if we are improving fitness through iterations:
        if self.importance == 0:
            if abs(self.fitness - self.last_fitness) <= self.epsilon_1:
                self.stagnation += 1
                # If sapso can't improve fitness within c_max iterations:
                # then importance is 1 (particle will go onto the best global
                # instead of gradient information)
                if self.stagnation == self.c_max:
                    self.importance = 1
                    self.stagnation = 0
            else:
                self.stagnation = 0

        elif self.importance == 1:
            if np.sqrt(np.sum(np.power((self.position - best_position), 2))) < self.epsilon_2:
                self.importance = 0
                self.stagnation = 0

                
    def adjust_direction(self, diversity):
        ''' Calculates direction of swarm's movements'''
        if (self.direction > 0 and diversity < self.d_low):  # must repulse
            self.direction = -1
            self.importance = 1
        elif (self.direction < 0 and diversity > self.d_high):  # must attract
            self.direction = 1
            self.importance = 0


##############################################################################################################


def initiate_swarm(n, n_dims, max_, min_, function, v_max, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high):
    ''' Initiate a swarm (group of particles) '''
    swarm = []
    for _ in range(n):
        position = np.array([(min_+ np.random.uniform() * (max_ - min_)) for _ in range(n_dims)])
        velocity = np.array([0.]*n_dims)
        importance = 1
        direction = 1
        stagnation = 0
        particle = Particle(position, velocity, function, importance, direction, stagnation, v_max, n_dims, min_, max_, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high)
        swarm.append(particle)
    return swarm


def evaluate_swarm(swarm):
    '''This method evaluates a swarm of particle.
        The process of evaluation includes sharing the best particle in the swarm
        every few iterations with all the other swarm running in parallel.
        At the end, the best particle found is save in a shared list of best particles.
    '''
    # Befores starting the optmization process, find the very first best position among particles
    fitness_list = np.array([particle.fitness for particle in swarm])
    best_fitness = np.amin(fitness_list)
    index = np.where(fitness_list == best_fitness)[0][0]
    best_position = swarm[index].position
    stagnation = 0
    for i in range(n_iters):
        #check if a new global_best was found in other swarms:
        if (i % migration_interval == 0):
            with acess_shared_info_lock:
                if shared_best_fitness.value < best_fitness:
                    best_fitness = shared_best_fitness.value
                    best_position = shared_best_position.value
                    #print('process {} recieved a new_best_from another swarm'.format(current_process().name))
            
        #sapso information:
        diversity = calculate_diversity(swarm, n, L)
        inertia = (max_inertia - i) * z
        last_best_fitness = best_fitness
        
        # classical pso information:
        for particle in swarm:
            particle.adjust_direction(diversity)
            particle.update_gradient()
            particle.update_velocity(inertia, best_position)
            particle.update_position()
            particle.update_fitness()
            # update global best:
            if particle.fitness < best_fitness:
                best_fitness = particle.fitness
                best_position = particle.position
                
            particle.update_importance(best_position)

        # Eventually let every other swarm know that a new global best was found:
        with acess_shared_info_lock:
            if best_fitness < shared_best_fitness.value:
                shared_best_fitness.value = best_fitness
                shared_best_position.value= best_position
                #print('process {} updated the global_best '.format(current_process().name))
        
        # Check for stop condition:
        stagnation += 1 if stop_condition(best_fitness, last_best_fitness, stop) else 0
        if stagnation >= stagnation_limit: 
            break

    # A swarm is finished by saving it's best particle in the shared list of best results
    with acess_shared_info_lock:
        shared_all_bests.append((best_fitness,best_position))

    return swarm


def calculate_diversity(swarm, n, L):
    """ Calcualte swarm diversity"""
    swarm_position = np.array([particle.position for particle in swarm])
    mean = np.mean(swarm_position, axis=0)
    minus_mean = np.array([particle - mean for particle in swarm_position]) ** 2
    factor = np.sum(np.sqrt(np.sum(minus_mean, axis=1)))
    diversity = 1. / (n * L) * factor
    return diversity


def stop_condition(best_fitness,last_best_fitness,stop): 
    ''' Check if no improvement in optmization was archieved.
    If so, return true to increment stagnation counter. '''   
    if np.all(abs(best_fitness - last_best_fitness) <= stop):
        return True
    return False


def make_global(n_particles, n_iterations, n_dims, diagonal_length, maximum_inertia, z_component, minimum_improvement, best_fitness, best_position, all_bests, acess_info_lock, limit, mi):
    ''' Python currently has a limitation in passing initial arguements to a pool of workers
        The workaround is to set global variables, with an initializer method, that will further
        pass all those variables to the workers in the pool. 
        When the pool is finished all these global variables will be deleted atomatically.
        Even though this is no good practice, it's unavoidable as there's no other solution'''

    global shared_best_fitness, shared_best_position, acess_shared_info_lock, shared_all_bests
    acess_shared_info_lock = acess_info_lock
    shared_best_fitness = best_fitness
    shared_best_position = best_position
    shared_all_bests = all_bests
    
    global stagnation_limit, migration_interval
    migration_interval = mi
    stagnation_limit = limit 
    
    global n, max_inertia, z, n_iters, stop, L
    n = n_particles
    max_inertia = maximum_inertia
    z = z_component
    n_iters = n_iterations
    stop = minimum_improvement
    L = diagonal_length


class Bunch(object):
    '''Safelly pass a bunch of parameters passed to the optmizer via the 'parameters' dict
     to the local namespace as individual variables'''
    def __init__(self, parameters):
        self.__dict__.update(parameters)