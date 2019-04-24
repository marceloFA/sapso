## Python SAPSO
Python SAPSO algorithm implemetation

### Description of files:
- *brute_force_sapso.py*:

  Technique of running executing SAPSO linearly in multiple cpu cores (brute-force approach)

- *numba_sapso.py*:

  SAPSO basic implementation using [numba](https://github.com/numba/numba) JIT compiler to improve performance

- *pathos_psapso.py*:

  Parallelized SAPSO using [pathos](https://github.com/uqfoundation/pathos) module for better multiprocessing information retrieving

- *sapso.py*:

   SAPSO basic implementation
- *test_functions.py*:
   
   A module composed of test functions for SAPSO performance tests, including funcitons minimuns and search space

- *test_parameters.py*:
	
	An example on how to build and *parameters* dictionary to be passed as the optmizer parameters of optmization

- *using_sapso.py*:

  An example of usage of the optmizer

- *using_sapso.py.lprof*:

  Binary file contaning a [line_profiler](https://github.com/rkern/line_profiler) (detailed information on execution time per line) of the optmizer main function (for execution time improvement purposes)

