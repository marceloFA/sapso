# -*- coding: utf-8 -*-
import numpy as np
from test_functions import TestFunctions
from os import getpid 
import ipdb # ipython debugger
from numba import jit
from typing import Dict

def calculate_diversity(swarm, L, n):
  """ Calculates dieversity of the swarm
    swarm = gorup of particles (n x n_dimensions)
    L = diagonal length of the search space (scalar)
    n = number of particles (scalar)
    does not support numba (kwarg of np.mean is not supported)
  """
  sum_ = 0
  mean = np.mean(swarm, axis=0)
  minus_mean = np.array([particle - mean for particle in swarm]) ** 2
  factor = np.sum(np.sqrt(np.sum(minus_mean,axis=1)))
  diversity = 1./(n*L) * factor
  return diversity


@jit(nopython=True)
def calculate_dir_and_importance(dir_, diversity, I, n, d_low, d_high):
  ''' Calculates direction and importance
  Supports  numba'''
  if (dir_ > 0 and diversity < d_low): # must repulse
    dir_ = -1
    I = np.ones(n)
  elif (dir_ < 0 and diversity > d_high): # must attract
    dir_ = 1
    I = np.zeros(n)

  return dir_, I



def calculate_gradient(f, particle,last_fitness):
  ''' Doc string 
    Supports numba (make execution way longer, tough) but nopython is not supported'''
  gradient = list()
  step = 1e-5
  f_p = last_fitness

  for i in range(len(particle)):
    xl = particle
    xl[i] += step
    gradient.append( (f(xl) - f_p)/step)
  return gradient

@jit(nopython=True)
def validate_gradient(gradient,max_value):
  '''Validates velocitites based on a maximum speed factor '''
  gradient[gradient > max_value] = max_value
  gradient[gradient < -max_value] = -max_value
  return gradient


@jit(nopython=True)
def calculate_velocity(velocity, particle, importance, gradient, inertia, dir_, c1, c2, n_dimensions, best_global_position):
  ''' Calculates swarm velocity
      Supports numba '''
  phi_1 = np.random.standard_normal(size=n_dimensions)
  phi_2 = np.random.standard_normal(size=n_dimensions)
  return (inertia*velocity) + dir_ * ( (importance * c1 * phi_1 *(best_global_position - particle) + (importance-1)* c2 * phi_2 *gradient) )
  

@jit(nopython=True)
def validate_velocity(velocity, max_velocity):
  ''' Validates velocitites based on a maximum speed factor
      Supports numba '''
  velocity[ velocity > max_velocity] = max_velocity
  velocity[ velocity < -max_velocity] = -max_velocity
  return velocity


@jit(nopython=True)
def update_position(particle,velocity):
  ''' updates all particle's positions based on their velocity
  Supports numba '''
  particle += velocity
  return particle

@jit(nopython=True)
def validate_position(particle, importance, c, max_,min_):
  '''Validates new particle position based on the search space limits '''
  for k in range(len(particle)):
    if particle[k] < min_:
      particle[k] = min_
      importance = 1
      c = 0

    elif particle[k] > max_:
      particle[k] = max_
      importance = 1
      c = 0
  return particle, importance, c


def calculate_fitness(objective_function, particle):
  ''' Calculates Fitness (Y = f(x)) based on the objective function
  Does not upports numba.
  Following test functions were tested with numba:
  Sphere
  '''
  return objective_function(particle)

@jit(nopython=True)
def update_best_global(particle, fitness, best_global_fitness, best_global_position):
  '''After an iteration the best fitness found must be updated
  Supports numba'''
  new_best_global_fitness = np.float64(0)
  new_best_global_position = np.array([0],dtype=np.float64)

  if fitness < best_global_fitness:
    new_best_global_fitness = fitness
    new_best_global_position = particle

  return new_best_global_fitness, new_best_global_position

@jit(nopython=True)
def update_importance(I, swarm, fitness, last_fitness, best_global_position, counter, epsilon, epsilon_2, c_max, n):
  '''After an iteration importance for each particle must be updated'''
  for k in range(n):
    #Check to see if we are improving fitness through iterations:
    if I[k] == 0:
      if abs(fitness[k] - last_fitness[k]) <= epsilon:
        counter[k] += 1
        #If sapso can't improve fitness within c_max iterations:
        # then importance is 1 (particle will go onto the best global instead of gradient information)
        if counter[k] == c_max:
          I[k] = 1
          counter[k] = 0

      else:
        counter[k] = 0

    elif I[k] == 1:
      if np.sqrt(np.sum(np.power((swarm[k] - best_global_position),2))) < epsilon_2:
        I[k] = 0
        counter[k] = 0

  return I, counter

@profile
def sapso(parameters):
  ''' The Semi Autonomus particle swarm optmizer '''
  # Parameters as individual variables:
  p = Bunch(parameters)
  n = p.n
  m = p.m
  stop_criterion = p.stop_criterion
  n_dimensions = p.n_dimensions
  min_inertia = p.min_inertia
  max_inertia = p.max_inertia
  c1 = p.c1
  c2 = p.c2
  c_max = p.c_max
  epsilon = p.epsilon
  d_low = p.d_low
  d_high = p.d_high
  f_name = p.f_name
  return_dict = p.return_dict


  pid = getpid()
  min_ , max_ = getattr(TestFunctions(),f_name+'_space') # Search space limitation
  z = (max_inertia - min_inertia)/m               # inertia component
  velocity = np.zeros((n,n_dimensions))           # Particle's velocity
  gradient = np.zeros((n,n_dimensions))           # Particle's gradient information
  v_max = abs(max_ - min_)/2                      # Maximum velocity
  counter = np.zeros(n)                           # Responsible for changing the I variable state (esse contador é quem mostrará o momento de trocar para a componente social ou gradient)
  dir_ = 1                                        # Direction [1 (attraction) or -1 (repulsion)]
  L = np.linalg.norm([max_-min_ for _ in range(n_dimensions)])  # Maximum radius of the search space
  diversity = 0.                                  # Diversity factor
  best_fitness_history = []                       # Best fitness history (for each iteration)
  objective_function = getattr(TestFunctions(),f_name)
  epsilon_2 = 1e-5                                # how does the second epsilon works?

  # Initializing ('iteration 0'):
  #Start swarm's particles at a random location:
  swarm = np.array([ [min_ + np.random.uniform()*(max_-min_) for i in range(n_dimensions)] for _ in range(n)])

  # Importance (starts as '1' [attraction phase] by default):
  I = np.ones(n)
  # Initiate fitness list:
  current_fitness = np.array(list(map(objective_function,swarm)))
  
  #Initiate best global fitness:
  best_global_fitness = np.amin(current_fitness)
  # Initiate best global position:
  best_global_position = swarm[np.where(current_fitness == best_global_fitness)][0]
  # Main loop:
  for i in range(m):
      # Save last iteration fitness:
      last_fitness = current_fitness
      #Calculate inertia as a function of remaining iterations:
      inertia = (max_inertia - i) * z
      #print('iteration'+str(i))

      for k in range(n):
          #Calculate Gradient:
          gradient[k] = calculate_gradient(objective_function, swarm[k],last_fitness[k])
          gradient[k] = validate_gradient(gradient[k], v_max)

          #Calculate Velocity:
          velocity[k] = calculate_velocity(velocity[k], swarm[k], I[k], gradient[k], inertia, dir_, c1, c2, n_dimensions,best_global_position)
          velocity[k] = validate_velocity(velocity[k], v_max)

          # Update Positions:
          swarm[k] = update_position(swarm[k], velocity[k])
          swarm[k], I[k], counter[k] = validate_position(swarm[k], I[k], counter[k], max_, min_)

          #Update Fitness list:
          current_fitness[k] = objective_function(swarm[k])

          #Update best global position and fitness:
          best_global_fitness, best_global_position = update_best_global(swarm[k], current_fitness[k], best_global_fitness, best_global_position)

      #Update importance:
      I, counter = update_importance(I, swarm, current_fitness, last_fitness, best_global_position, counter, epsilon, epsilon_2, c_max, n)

      #Recalculate diversity and direction:
      diversity = calculate_diversity(swarm, L, n)
      dir_, I   = calculate_dir_and_importance(dir_, diversity, I, n, d_low, d_high)
    
  
  # Only for parallel executions
  if return_dict:
    return_dict[pid] = [best_global_position, best_global_fitness]
    return return_dict

  return best_global_position, best_global_fitness


class Bunch(object):
  ''' Import parameters dicitionary into sapso script namespace'''
  def __init__(self, parameters):
    self.__dict__.update(parameters)