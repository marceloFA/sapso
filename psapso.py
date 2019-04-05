import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

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


def calculate_gradient(objective_function, particle, n_dimensions):
  gradient = [calculate_partial_derivative(objective_function, particle, i) for i in range(n_dimensions)]
  return np.array(gradient)

def validate_gradient(gradient,max_value):
  for v in gradient:
    if v < -max_value:  v = -max_value
    elif v > max_value: v = max_value 

  return gradient

def plot_swarm(swarm):
  '''
    Plot swarm movimentation through the search space
  '''
  x = [position[0] for position in swarm]
  y = [position[1] for position in swarm]
  plt.scatter(x, y)


