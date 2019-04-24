from sapso import sapso
from test_functions import TestFunctions
from multiprocessing import Process, Manager
from test_parameters import parameters


print(parameters)

n_processes = 4
# A list to hold all processes:
processess_list = []
# A process manager to get results of executions back:
manager = Manager()
return_dict = manager.dict()

# Start every process:
for n in range(n_processes):
        process = Process(target=sapso, args=(parameters))
        processess_list.append(process)
        process.start()

# Join then once it's finished:
for process in processess_list:
	process.join()

#Initial tests:
print (return_dict.values())


