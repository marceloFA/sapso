# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TestFunctions():
  
  def rosenbrock(self,x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


  def sphere(self,x):
    """The Sphere function"""
    return sum(i*i for i in x)


  def rastrigin(self,x):
    """The Rastrigin function"""
    return 10 * len(x) + sum(i * i - 10 * np.cos(2 * np.pi * i) for i in x)


  def himmelblau(self,x):
    """The Himmelblau function"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


  def ackley(self,x):
    """The Ackley function"""
    dim = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(dim):
        sum1 += x[i]**2
        sum2 += np.cos(2*np.pi*x[i])

    return -20*np.exp(-0.2*np.sqrt(sum1/dim)) - np.exp(sum2/dim) + 20 + np.e


  def levi_13(self,x):
    """The Levy Function N. 13"""
    return np.sin(3*np.pi*x[0])**2 + (x[0] - 1)**2*(1 + np.sin(3*np.pi*x[1])**2) + (x[1] - 1)**2*(1 + np.sin(2*np.pi*x[1])**2)


  def matyas(self,x):
    """The Matyas function"""
    return (0.26 * (x[0]**2 + x[1]**2)) - (0.48 * x[0] * x[1])


  def booth(self,x):
    """The Booth function"""
    return ((x[0] + 2*x[1] - 7)**2) + ((2*x[0] + x[1] +-5)**2)
  
  def beale(self,x):
    """ The Beale function """
    return (1.5-x[0]+(x[0]*x[1]))**2 + (2.25-x[0]*x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2



def calculate_diversity_and_dir(d_low, d_high, n, L, swarm, dir_):
  """  Calculates the diversity factor, 
       which may be -1 (repislonulsion phase) or 1 (attraction phase)
  """
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


# TODO: refetorar com map()
#Entender melhor como funcionam as derivadas parciais
def calculate_partial_derivative(objective_function, particle, i):
  h = 1e-5
  x = particle
  xi = x[i]
  
  x[i] = xi + h
  func_plus_h = objective_function(x)

  x[i] = xi - h
  func_minus_h = objective_function(x)

  return (func_plus_h - func_minus_h) / (2*h)


# Calculates the gradient:
def calculate_gradient(objective_function, particle, n_dimensions):
    gradient = [calculate_partial_derivative(objective_function, particle, i) for i in range(n_dimensions)]
    return np.array(gradient)


def plot_swarm(swarm):
  '''
    Plot swarm movimentation through the search space
  '''
  x = [position[0] for position in swarm]
  y = [position[1] for position in swarm]
  plt.scatter(x, y)



def sapso(n, m, n_dimensions, min_, max_, min_inertia, max_inertia, c1, c2, c_max, d_low, d_high, epislon, f_name, stop_criterion):
  #TODO: docstring later on
  '''
    The Semi Autonomus particle swarm optmizer
  '''
  #fig = plt.figure(figsize=(9,9))                 # Object for ploting
  #plt.title('Particles moving through search space')
  
  swarm = np.zeros((n, n_dimensions))             # Current position of all swarm's particles
 
  velocities = np.zeros((n, n_dimensions))        # Particle's velocities    
  
  v_max = abs(max_ - min_)/2                      # Maximum velocities
 
  best_global_position = np.zeros((n_dimensions)) # Memory of best ever found position
 
  fitness_list = np.zeros(n)                      # A list of current fitness o evey particle. reseted every cycle

  old_fitness_list = np.zeros(n)                  # A list of past iteration fitness o evey particle. reseted every cycle

  best_global_fitness = 0.0                       # Memory of best ever found fitness
  
  I = np.zeros((n))                               # Variável para decidir qual dos componentes da equação usar (Preciso entender melhor essa parte)
    
  counter = np.zeros((n))                         # Responsible for changing the I variable state (esse contador é quem mostrará o momento de trocar para a componente social ou gradient)
    
  dir_ = 1                                        # Direction [1 (attraction) or -1 (repulsion)]

  L = abs(max_-min_)                              # Maximum radius of the search space
  
  diversity = 0.                                  # Diversity factor

  best_fitness_history = []                       # Best fitness history (for each iteration)

  objective_function = getattr(TestFunctions(),f_name)

  epsilon_2 = 1e-5                                # how does the second epsilon works?
  
  # For each particle:
  for k in range(n):
      #Start it at a random location:
      swarm[k] = np.array([min_ + np.random.uniform()*(max_-min_) for i in range(n_dimensions)])

  # Initiate best fitness:
  fitness_list = list(map(objective_function,swarm))
  #Initiate best global fitness:
  best_global_fitness = min(fitness_list)
  # Initiate best global position using positions of the first swarm's particle:
  best_global_position = swarm[fitness_list.index(best_global_fitness)]

  
  # Main loop:
  for i in range(m):
      #Calculate inertia as a function of remaining iterations:
      inertia = (min_inertia - (min_inertia - max_inertia)) * (i/m); 
      
      #For each particle:
      for k in range(n):
          # Reset new_position and gradient information:
          new_position = np.zeros((n_dimensions))
          gradient = np.zeros((n_dimensions))
          
          #If particle is following gradient information,calculate it:
          if I[k] == 0:
              gradient = calculate_gradient(objective_function, swarm[k], n_dimensions)
          
          # Calculate particle velocity:
          phi_1 = np.random.uniform(size=n_dimensions)
          phi_2 = np.random.uniform(size=n_dimensions)
          velocity = (inertia*velocities[k]) + dir_ * (I[k]*c1*phi_1*(best_global_position - swarm[k]) + (I[k]-1)*c2*phi_2*gradient)

          #For each dimension:
          for j in range(n_dimensions):
              #Validate velocity:
              if velocity[j] > v_max: velocity[j] = v_max
              if velocity[j] < -v_max: velocity[j] = -v_max
              
              #Update position:
              new_position[j] = swarm[k][j] + velocity[j]

              # Validate position: # TOOD: Reffactor to a simpler sintax
              if new_position[j] > max_: 
                new_position[j] = max_
                I[k] == 1
                counter[k] == 0 

              if new_position[j] < min_:
                new_position[j] = min_
                I[k] == 1
                counter[k] == 0
          
          swarm[k] = new_position
          velocities[k] = velocity
          fitness = objective_function(swarm[k])

           #Verify if it's a new global best:
          if fitness < best_global_fitness:
              best_global_fitness = fitness
              best_global_position = new_position
          
          fitness_list[k] = fitness
          old_fitness_list = fitness_list

      # if I = 0 particle is following gradient information
      # if I = 1 particle is following global best information
      #differently from other pso version, in sapso a particle can either follow one or another direction (in addition to alway following )
      
      # After moving all particles, update their I value:
      for k in range(n):    
        #Check to see if we are improving fitness through iterations:
        if I[k] == 0:
          if abs(fitness_list[k] - old_fitness_list[k]) <= epislon:
            counter[k] += 1
            #If SAPSO can not improve fitness in 'c_max' iterations: then 'I' is 1 (particle will go onto the best global instead of gradient information)
            if counter[k] == c_max:
              I[k] = 1
              counter[k] = 0
                  
          else:
            counter[k] = 0
          
        if I[k] == 1:
          if abs(np.sqrt(np.sum((swarm[k] - best_global_position)**2))) < epsilon_2:
            I[k] = 0
            counter[k] = 0
      
      #Recalculate diversity:
      diversity, dir_ = calculate_diversity_and_dir(d_low, d_high, n, L, swarm, dir_)
      #
      best_fitness_history.append(best_global_fitness)
      #Stop criterion:
      #if len(best_fitness_history) >2 and abs(best_fitness_history[-1]-best_fitness_history[-2]) <= stop_criterion: break
  #recreating the fig object for new and final plot:
  fig = plt.figure(figsize=(9,9))
  plt.title('Final Position of particles')
  plot_swarm(swarm)
  final_positions_plot = 'final_positions_plot.png'
  plt.savefig(final_positions_plot)
  return final_positions_plot, best_fitness_history, best_global_position