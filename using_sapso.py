# -*- coding: utf-8 -*-
from sapso import sapso
from PIL import Image
## Parameters 
n = 80                 # Particles
m = 5000               # Iterations
n_dimensions = 2       # Number of dimensions
min_ = -5.             # Search space size
max_ = min_ * -1       # Seach space size  

min_inertia = .4       # Inertial weight
max_inertia = .7       # Inertial weight
c1 = 1.5               # Cognitive coefficient
c2 = 1.5               # Social coefficient
c_max = 3              # Max consecutive evaluations before finish optmizing
stop_criterion = 1e-10 # If the algorithm cannot improve fitness it must stop
d_low = 1e-1           # Lower threshold for diversity control
d_high = .25          # Upper threshold for diversity control

f_name = 'ackley'   # Select your optmization


#Using the optmizer:
final_positions, best_fitness_history, best_global_position = sapso(n, m, n_dimensions, min_, max_, min_inertia, max_inertia, c1, c2, c_max, d_low, d_high, stop_criterion, f_name)
global_minimum = [0,0]

print('Best fitness found was {} and actual global minimum is{}'.format(best_global_position,global_minimum))
print('Particles\' last positions were: ')
#...

img = Image.open(final_positions)
img.show()
#Destroy plot after execution:
from os import remove
remove(final_positions)