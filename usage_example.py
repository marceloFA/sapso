from time import time
# Local imports
from sapso.optmizer import sapso
from psapso.optmizer import parallel_sapso
from test_functions import TestFunctions
from example_parameters import parameters


# Sequential sapso:
start = time()
best_position, best_fitness = sapso(parameters)
finish = time() - start
print('Execution time was: ',finish,' seconds')
print('Best found was:\n',best_fitness,best_position)


# Parallel sapso:
start = time()
all_bests, results = parallel_sapso(parameters)
finish = time() - start
print('Execution time was: ',finish,' seconds')
print('Array of all best results is:')    
for best in all_bests:
        print('\n',best[0],'\n',best[1])
