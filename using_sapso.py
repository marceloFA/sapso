# -*- coding: utf-8 -*-
from sapso import sapso
from test_functions import TestFunctions
## Parameters
n = 20                 # Particles
m = 1000               # Iterations
stop_criterion = 1e-10 # Stop criterion
n_dimensions = 2       # Number of dimensions     
min_inertia = .9       # Inertial weight
max_inertia = .4       # Inertial weight
c1 = 2                 # Cognitive coefficient
c2 = 2                 # Social coefficient
c_max = 3              # Max consecutive evaluations before finish optmizing
epsilon = 1e-2         # If the algorithm cannot improve fitness it must stop
d_low = 1e-1           # Lower threshold for diversity control
d_high = .25           # Upper threshold for diversity control
f_name = 'rosenbrock'  # Select your optmization function


#Using the optmizer:
position, minimum_found = sapso(n, m, n_dimensions, min_inertia, max_inertia, c1, c2, c_max, d_low, d_high, epsilon, f_name, stop_criterion)
real_minimum = getattr(TestFunctions(),f_name+'_min')
f = getattr(TestFunctions(),f_name)
print('Best fitness found was {} and actual global minimum is {}'.format(minimum_found, f(real_minimum)))
print('Position was {} and actual global minimum position is {}'.format(position, real_minimum))
