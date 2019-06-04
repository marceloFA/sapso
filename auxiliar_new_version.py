import numpy as np
from multiprocessing import Queue, Process

class Particle():
    ''' This class defines a particle.
        An instance of a particle contains fundamental information for psapso execution '''
    
    def __init__(self, position, velocity, function, importance, direction, stagnation, v_max, n_dims, min_, max_, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high):
        ''' Initiates a particle '''
        self.position = position
        self.velocity = velocity
        self.objective_function = function
        self.fitness = self.objective_function(self.position)
        self.last_fitness = np.copy(self.fitness)
        self.gradient = np.array([0.]*n_dims) # is it right?
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
        ''' Take (a lot) of parameters and calculates particle's velocity '''
        
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
        ''' Calculates gradient of a particle '''
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
        '''Updates a particle's position '''
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
        ''' Updates the importance factor'''
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
# Global functions:

def initiate_swarm(n, n_dims, max_, min_, function, v_max, c1, c2, epsilon_1, epsilon_2, c_max, d_low, d_high):
    ''' Initiate a swarm (group of particles)'''
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


def calculate_diversity(swarm, n, L):
    """ Calculates dieversity of the swarm"""
    swarm_position = np.array([particle.position for particle in swarm])
    mean = np.mean(swarm_position, axis=0)
    minus_mean = np.array([particle - mean for particle in swarm_position]) ** 2
    factor = np.sum(np.sqrt(np.sum(minus_mean, axis=1)))
    diversity = 1. / (n * L) * factor
    return diversity


def evaluate_particle():
    '''Will be passed to a process for single avaliation. concurrency is archieved through this'''


#############################################################################################
# Helper functions
class Bunch(object):
    '''Saves parameters to the local namespace as individual variables'''

    def __init__(self, parameters):
        self.__dict__.update(parameters)

def stop_condition(best_fitness,last_best_fitness,stop):    
    if np.all(abs(best_fitness - last_best_fitness) <= stop):
        return True
    return False