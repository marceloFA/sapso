# -*- coding: utf-8 -*-
from sapso import sapso
from test_functions import TestFunctions
from time import process_time
from test_parameters import parameters

#Using the optmizer:
begin = process_time()
position, minimum_found = sapso(parameters)
time_elapsed = process_time()-begin

f_name = parameters['f_name']
real_minimum = getattr(TestFunctions(),f_name+'_min')
f = getattr(TestFunctions(),f_name)
print('Best fitness found was {} and actual global minimum is {}'.format(minimum_found, f(real_minimum)))
print('Position was {} and actual global minimum position is {}'.format(position, real_minimum))
print('Execution time was ',time_elapsed,'seconds')

