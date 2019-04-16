#from pathos.multiprocessing import ProcessingPool as Pool
from pathos import multiprocessing as mp
from pprint import pprint
from test_parameters import parameters
from sapso import sapso

n_iterations = 10
results = mp.ProcessingPool().map(sapso, [parameters]*n_iterations)
pprint(results)