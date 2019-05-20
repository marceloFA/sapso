# -*- coding: utf-8 -*-
from sapso import sapso # not parallel
#from psapso import sapso # parallel
from test_functions import TestFunctions

f_name = 'sphere'
parallel = True
parameters = (20, 1000, 2, .9, .4, 2, 2, 3, 1e-2, 1e-1, .25, 1e-10, f_name, parallel)

# Using the optmizer:
position, minimum_found = sapso(parameters)

# Printing out results:
real_minimum = getattr(TestFunctions(), f_name + '_min')
f = getattr(TestFunctions(), f_name)

print('Best fitness found was {} and actual global minimum is {}'.format(
    minimum_found, f(real_minimum)))
print('Position was {} and actual global minimum position is {}'.format(
    position, real_minimum))

