from new_version import pso
from test_functions import TestFunctions
from test_parameters import parameters
from time import time

# Using the optmizer:
f_name = parameters['f_name']
start = time()
results = pso(parameters)
print(results)
finish = time() - start
# Printing out results:
#real_minimum = getattr(TestFunctions(), f_name + '_min')
#f = getattr(TestFunctions(), f_name)
 
#print('Best fitness found was {} and actual global minimum is {}'.format(
#    minimum_found, f(real_minimum)))
#print('Position was {} and actual global minimum position is {}'.format(
#    position, real_minimum))
print('Execution time was: ',finish,' seconds')
#print('Finished at iteration {i}'.format(i=i))
#print('Grad info was not calculated, as it was not needed in {} iterations'.format(grad_skip_count))
