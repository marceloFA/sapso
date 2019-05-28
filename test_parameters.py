
parameters = {
	'n' : 20,                  # Particles
	'm' : 1000 ,               # Iterations
	'stop' : 1e-10,            # Stop criterion
	'n_dimensions' : 2,        # Number of dimensions     
	'min_inertia' : .9,        # Inertial weight
	'max_inertia' : .4,        # Inertial weight
	'c1' : 2,                  # Cognitive coefficient
	'c2' : 2,                  # Social coefficient
	'c_max' : 3,               # Max consecutive evaluations before finish optmizing
	'epsilon' : 1e-2,          # If the algorithm cannot improve fitness it must stop
	'd_low' : 1e-1,            # Lower threshold for diversity control
	'd_high' : .25,            # Upper threshold for diversity control
	'f_name' : 'rastrigin',    # Select your optmization function
	'parallel': True,          # Defines if parallel gradient calculation happens
	}