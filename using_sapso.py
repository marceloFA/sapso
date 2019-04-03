# -*- coding: utf-8 -*-
from sapso import sapso

## Parameters
n = 50                # Particles
m = 3000              # Iterations
n_dimensions = 2      # Number of dimensions
min_ = -50.           # Search space size
max_ = min_ * -1      # Seach space size  

initial_inertia = 0.4 # Inertial weight
final_inertia = 0.7   # Inertial weight
c1 = 1.5              # Cognitive coefficient
c2 = 1.5              # Social coefficient
c_max= 3              # Max consecutive evaluations before finish optmizing

d_low = 1e-6          # Lower threshold for diversity control
d_high = 0.25         # Upper threshold for diversity control

f_name = 'rastrigin'    # Select your optmization


#Using the optmizer:
best_fitness_history, best_global_position = sapso(n, m, n_dimensions, min_, max_, initial_inertia, final_inertia, c1, c2, c_max, d_low, d_high, f_name)
global_minimum = [0,0]

print('Best fitness found was {} \n and actual global minimum is{}'.format(best_global_position,global_minimum))
print('difference is: {}'.format(best_global_position-global_minimum))