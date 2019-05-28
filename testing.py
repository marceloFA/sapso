from time import time
from sapso import sapso
n = 20;m = 5000;n_dims = 2;min_inertia = .9;max_inertia = .4
c1 = c2 = 2;c_max = 3;d_low = .25;d_high = 1e-10
epsilon = 1e-2;stop = 1e-10;f_name = 'rastrigin'

parallel = True
parallel_results = {}
values_for_testing = [2]
repeat = 1

parameters = (n, m, n_dims, min_inertia, max_inertia, c1, c2, c_max, epsilon, d_low, d_high, stop, f_name, parallel)

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