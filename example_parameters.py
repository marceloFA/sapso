parameters = {
	'n' : 50,                     # Particles
	'm' : 5000,                    # Iterations
    'n_swarms': 5,                # Number of swarm (parallel execution)
    'migration_interval': 20,      # Exchange best found every few iterations
	'stagnation_limit': 500,       # Number of consecutive stagnations before stop
 	'minimum_improvement' : 1e-10, # Below this stagnaiton condition is triggered
	'n_dimensions' : 2,            # Number of dimensions     
	'min_inertia' : .9,            # Inertial weight
	'max_inertia' : .4,            # Inertial weight
	'c1' : 2,                      # Cognitive coefficient
	'c2' : 2,                  	   # Social coefficient
	'c_max' : 3,               	   # Max consecutive evaluations before finish optmizing
	'epsilon' : 1e-2,          	   # If the algorithm cannot improve fitness it must stop
	'd_low' : .2,    	           # Lower threshold for diversity control
	'd_high' : .225,               # Upper threshold for diversity control
	'f_name' : 'sphere',       # Select your optmization function
	'parallel_grad': True          # Whether gradient will be calculated in parallel (sequential sapso)
	}
