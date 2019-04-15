from sapso import sapso
from test_functions import TestFunctions
from multiprocessing import Process, Manager
## Parameters
n = 20                 # Particles
m = 5000               # Iterations
stop_criterion = 1e-10 # Stop criterion
n_dimensions = 50       # Number of dimensions     
min_inertia = .9       # Inertial weight
max_inertia = .4       # Inertial weight
c1 = 2                 # Cognitive coefficient
c2 = 2                 # Social coefficient
c_max = 3              # Max consecutive evaluations before finish optmizing
epsilon = 1e-2         # If the algorithm cannot improve fitness it must stop
d_low = 1e-1           # Lower threshold for diversity control
d_high = .25           # Upper threshold for diversity control
f_name = 'sphere'  # Select your optmization function




n_processes = 4
# A list to hold all processes:
processess_list = []
# A process manager to get results of executions back:
manager = Manager()
return_dict = manager.dict()

# Start every process:
for n in range(n_processes):
        process = Process(target=sapso, args=(n, m, n_dimensions, min_inertia, 
											  max_inertia, c1, c2, c_max, d_low, 
											  d_high, epsilon, f_name, stop_criterion, return_dict))
        processess_list.append(process)
        process.start()

# Join then once it's finished:
for process in processess_list:
	process.join()

#Initial tests:
print (return_dict.values())

