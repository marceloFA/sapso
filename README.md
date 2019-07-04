## Python SAPSO and PSAPSO

Python implemetations (sequential and parallel) of the semi autonomous particle swarm optmizer ([sapso](https://www.sciencedirect.com/science/article/pii/S1568494618302187))


### Prerequisites

[numpy](https://github.com/numpy/numpy)


### Optmizing a function:

1- Define your parameters of optmization in a dictionary as specified in the [example](example_parameters.py)

2- Call the optmizer of your choice passing those parameters:

```
   from psapso.optmizer import sapso
   sapso(parameters)
```

### Description of files:

- */sapso*:

   SAPSO basic implementation, allows to chose between sequential and parallel gradient calculation

- */psapso*:

   Parallel SAPSO *beta* implementation

- */psapso slow_info_exchange*:

   Parallel SAPSO based on slow informaitone exchange *beta* implementation

- */scripts*:

   Scripts used to benchmark the optmizers

- *test_functions.py*:
   
   A module composed of mathematical functions for optmization tests

- *example_parameters.py*:
	
	An example on how to build a parameters dictionary that must be passed to the optmizer


- */scripts/example_file_name.py.lprof*:

  Binary file contaning a [line_profiler](https://github.com/rkern/line_profiler) (detailed information on execution time per line) of the optmizer main function (for execution time improvement purposes)


### Authors

* **Marcelo Freitas** - [marcelofa](https://github.com/marcelofa)

### Acknowledgments

* **Reginaldo Santos** - initial sapso paper and [implementation](https://github.com/regicsf2010/SAPSO)  
* **Abner Cardoso** - initial studies on parallel sapso and [implementation](https://bitbucket.org/abncardoso/psapso/src/master/)  
