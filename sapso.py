# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from test_functions import TestFunctions


def calculate_diversity_and_dir(d_low, d_high, n, L, swarm, dir_):
  """  Calculates the diversity factor, 
       which may be -1 (repislonulsion phase) or 1 (attraction phase) """
  mean = np.mean(swarm, axis=0)
  summ = 0
  for k in range(n):
    summ += np.sqrt( np.sum( swarm[k] - mean ) **2 )
  diversity = 1./(n*L) * summ
  
  if diversity < d_low:
    dir_ = -1
  elif diversity > d_high:
    dir_ = 1
  
  return diversity, dir_


def calculate_partial_derivative(objective_function, particle, i):
  h = 1e-5
  x = particle
  xi = x[i]
  
  x[i] = xi + h
  func_plus_h = objective_function(x)

  x[i] = xi - h
  func_minus_h = objective_function(x)

  return (func_plus_h - func_minus_h) / (2*h)


def calculate_velocities(previous_velocities,swarm,I,gradients,inertia,dir_,c1,c2):
  '''Calculates swarm velocities'''
  new_velocities = []
  phi_1 = np.random.uniform(size=n_dimensions)
  phi_2 = np.random.uniform(size=n_dimensions)
  for velocity, particle, importance, gradient in zip(previous_velocities,swarm,I, gradients):
    new_velocities.append( (inertia*velocity) + dir_ * (importance*c1*phi_1*(best_global_position - particle) + (importance-1)*c2*phi_2*gradient) )
  return np.array(new_velocities)


def validate_velocities(velocities,max_velocity):
  ''' Validates velocitites based on a maximum speed factor'''
  for velocity in velocities:
    for component in velocity:
      if component < -max_velocity:  component = -max_velocity
      elif component > max_velocity: component = max_velocity 
  return np.array(velocities)


def calculate_gradients(objective_function, swarm, n_dimensions):
  '''Calculates gradients'''
  gradients = []
  for particle in swarm:
    gradients.append([calculate_partial_derivative(objective_function, particle) for i in range(n_dimensions)])
  return np.array(gradients)


def validate_gradients(gradients,max_value):
  '''Validates velocitites based on a maximum speed factor '''
  for gradient in gradients:
    for v in gradient:
      if v < -max_value:  v = -max_value
      elif v > max_value: v = max_value 
  return np.array(gradient)


def update_positions(swarm,velocities):
  ''' updates all particle's positions based on their velocity '''
  for particle, velocity in zip(swarm, velocities):
    particle += velocity
  return np.array(swarm)


def validate_positions(swarm, max_,min_,I,counter):
  '''Validates new particle position based on the search space limits '''
  for particle, importance, c in zip(swarm,I,counter):
    for position in particle:
      if position < min_:  
        position = min_
        importance == 1
        c == 0 

      elif position > max_:
       position = max_ 
       importance == 1
       c == 0 
  return np.array(swarm)


def calculate_fitness(objective_function, swarm):
  ''' Calculates Fitness (Y = f(x)) based on the objective function'''
  fitness = []
  for particle in swarm:
    fitness.append(objective_function(particle))
  return np.array(fitness)

def update_best_global(swarm, best_global_fitness, best_global_position):
  '''After an iteration the best fitness found must be updated'''
  for particle, fit in zip(swarm,fitness):
    if fit < best_global_fitness:
      best_global_fitness = fit
      best_global_position = particle
  return best_global_fitness, best_global_position

def update_importance(I, fitness, last_fitness, counter, epislon, epislon_2, c_max):
  '''After an iteration importance for each particle must be updated'''
  for k in range(n):    
    #Check to see if we are improving fitness through iterations:
    if I[k] == 0:
      if abs(fitness_list[k] - last_fitness_list[k]) <= epislon:
        counter[k] += 1
        #If sapso can't improve fitness within c_max iterations:
        # then importance is 1 (particle will go onto the best global instead of gradient information)
        if counter[k] == c_max:
          I[k] = 1
          counter[k] = 0
              
      else:
        counter[k] = 0
      
    if I[k] == 1:
      if abs(np.sqrt(np.sum((swarm[k] - best_global_position)**2))) < epsilon_2:
        I[k] = 0
        counter[k] = 0

  return np.array(I)


def plot_swarm(swarm):
  ''' Plot swarm movimentation through the search space'''
  x = [position[0] for position in swarm]
  y = [position[1] for position in swarm]
  plt.scatter(x, y)


def sapso(n, m, n_dimensions, min_, max_, min_inertia, max_inertia, c1, c2, c_max, d_low, d_high, epislon, f_name, stop_criterion):
  ''' The Semi Autonomus particle swarm optmizer '''  

  z = (max_inertia - min_inertia)/m               # inertia component
 
  velocities = np.zeros((n, n_dimensions))        # Particle's velocities    
  
  v_max = abs(max_ - min_)/2                      # Maximum velocities
 
  best_global_position = np.zeros((n_dimensions)) # Memory of best ever found position
 
  fitness = np.zeros(n)                      # A list of current fitness o evey particle. reseted every cycle

  last_fitness = np.zeros(n)                  # A list of past iteration fitness o evey particle. reseted every cycle

  best_global_fitness = 0.0                       # Memory of best ever found fitness
  
  I = np.ones(n)                                  # Importance (starts as '1' [attraction phase] by default)
    
  counter = np.zeros(n)                           # Responsible for changing the I variable state (esse contador é quem mostrará o momento de trocar para a componente social ou gradient)
    
  dir_ = 1                                        # Direction [1 (attraction) or -1 (repulsion)]

  L = abs(max_-min_)                              # Maximum radius of the search space
  
  diversity = 0.                                  # Diversity factor

  best_fitness_history = []                       # Best fitness history (for each iteration)

  objective_function = getattr(TestFunctions(),f_name)

  epsilon_2 = 1e-5                                # how does the second epsilon works?
  
  # Initializing ('iteration 0'):
  #Start swarm's particles at a random location:
  swarm = np.array([ [min_ + np.random.uniform()*(max_-min_) for i in range(n_dimensions)] for _ in range(n)])

  # Initiate best fitness:
  fitness_list = list(map(objective_function,swarm))
  #Initiate best global fitness:
  best_global_fitness = min(fitness_list)
  # Initiate best global position using positions of the first swarm's particle:
  best_global_position = swarm[fitness_list.index(best_global_fitness)]

  
  # Main loop:
  for i in range(m):
      # Save last iteration fitness:
      last_fitness = fitness
      # Save last iteration velocities:
      previous_velocities = velocities
      #Calculate inertia as a function of remaining iterations:
      inertia = (max_inertia - m) * z 

      #Calculate Gradient:
      gradients = calculate_gradients(objective_function,swarm, n_dimensions)
      gradents = validate_gradients(gradients, v_max)

      # Calculate Velocity:
      velocities = calculate_velocities(previous_velocities, swarm, I, gradients, inertia, dir_, c1, c2)
      velocities = validate_velocities(velocities, v_max)

      # Update Positions:
      swarm = update_positions(swarm, veloctities)
      swarm = validate_positions(swarm, max_, min_, I, counter)

      #Update Fitness list:
      fitness = calculate_fitness(objective_function, swarm)

      best_global_fitness, best_global_position = update_best_global(swarm, best_global_fitness, best_global_position)

      I = update_importance(I, fitness, last_fitness, counter, epislon, epislon_2, c_max)
      
      #Recalculate diversity:
      diversity, dir_ = calculate_diversity_and_dir(d_low, d_high, n, L, swarm, dir_)
      
      best_fitness_history.append(best_global_fitness)
      #Stop criterion: TODO
      #if len(best_fitness_history) >2 and abs(best_fitness_history[-1]-best_fitness_history[-2]) <= stop_criterion: break
  
  return best_global_position