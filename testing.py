from time import time
from sapso import sapso
from test_parameters import parameters

parallel = True
parallel_results = {}
values_for_testing = [2]
repeat = 1

for value in values_for_testing:
    n_dims = value
    alist = []
    for i in range(repeat):
        start  = time()
        sapso(parameters)
        alist.append(time() - start)
    parallel_results[value] = alist
    print('finished for {} dimensions'.format(value))

print(parallel_results)