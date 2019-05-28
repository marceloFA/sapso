from sapso import sapso
from test_functions import TestFunctions
from test_parameters import parameters

# Using the optmizer:
f_name = parameters['f_name']
position, minimum_found = sapso(parameters)

# Printing out results:
real_minimum = getattr(TestFunctions(), f_name + '_min')
f = getattr(TestFunctions(), f_name)

print('Best fitness found was {} and actual global minimum is {}'.format(
    minimum_found, f(real_minimum)))
print('Position was {} and actual global minimum position is {}'.format(
    position, real_minimum))