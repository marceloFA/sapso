from sapso import sapso as pso
from test_functions import TestFunctions
from test_parameters import parameters
from time import time

# Using the optmizer:
f_name = parameters['f_name']
start = time()
position, minimum_found, i = pso(parameters)
finish = time() - start
# Printing out results:
real_minimum = getattr(TestFunctions(), f_name + '_min')
f = getattr(TestFunctions(), f_name)
 
print('Best fitness found was {} and actual global minimum is {}'.format(
    minimum_found, f(real_minimum)))
print('Position was {} and actual global minimum position is {}'.format(
    position, real_minimum))
print('Execution time was: ',finish,' seconds')
print('Finished at iteration {i}'.format(i=i))
