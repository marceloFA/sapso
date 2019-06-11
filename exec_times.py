from sys import argv
from time import time
from statistics import mean
# Local imports
from psapso.optmizer import parallel_sapso
from example_parameters import parameters
 
parameters['n_dims'] = argv[1]
parameters['n_sarms'] = argv[2]
f_name = ['sphere','rosenbrock','rastrigin','griewank','ackley','ellipsoid','alpine','bent_cigar','zakharov','happy_cat']
index = int(argv[3])
parameters['f_name'] = f_name[index]

start  = time()
best_particles = parallel_sapso(parameters)
exec_time = time() - start

best_fitness = [best[0] for best in best_particles] # get only the fitness

# formatted output to be written on the csv file:
print('{f_name},{n_dims},{n_swarms},{very_min},{mean_min},{exec_time},'.format(
    f_name = parameters['f_name'],
    n_dims = parameters['n_dims'],
    n_swarms = parameters['n_swarms'],
    very_min = min(best_fitness),
    mean_min = mean(best_fitness),
    exec_time = exec_time,
))