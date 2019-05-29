from psapso import psapso
from test_functions import TestFunctions
from test_parameters import parameters
from sys import argv
from time import time

parameters['n_dims'] = argv[1]
#parameters['parallel'] = True if argv[3] is '1' else False
f_name = ['rastrigin','sphere','rosenbrock']
index = int(argv[2])
parameters['f_name'] = f_name[index]
start  = time()
position, minimum_found = psapso(parameters)
exec_time = time() - start

# output
print('{f_name},{n_dims},{min_found},{n_iters},{exec_time},'.format(
    f_name = parameters['f_name'],
    n_dims = parameters['n_dims'],
    min_found = minimum_found,
    n_iters = parameters['m'],
    exec_time = exec_time,
))