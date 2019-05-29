for function in {0,1,2}; do
    for n_dims in {2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100}; do
        for n_execs in {1..30}; do
            python exec_times.py ${n_dims} ${function} >> exec_times.txt; 
            echo 'iteration '${n_execs}' ended';
        done
        echo 'Finished 30 execs for '${n_dims}' dims | function='${function};
    done
done

