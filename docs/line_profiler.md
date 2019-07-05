## Using line_profiler
*line_profiler* does line-by-line profiling of functions. This consists of getting detailed execution info of code.

kernprof is a convenient script for running either line_profiler or the Python standard library's cProfile or profile modules, depending on what is available.

To profile a script with *line_profiler* you must do the following:

### 1 - Install line profiler with pip 
```
pip install line_profiler
```
### 2 - run the line profiler
```
kernprof -l script.py
```
Where *script.py* is the script file name.
After profiling is finished a *.lprof* file is generated, it contains the script line profiling information
### 3- See results
```
python -m line_profiler script.py.lprof
```
This will display the line profling obtained from your script execution behavior

### 4- One command variation
In case you want to quickly execute a profiling and display it right after execution, *&&* will do the trick
```
kernprof -l script.py && python -m line_profiler script.py.lprof
```
### 5 - Aditional details
For any aditional detail on the *line_profiler* module, please refer to the original [documentation and repository](github.com/rkern/line_profiler)
