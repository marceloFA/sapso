for parallel in {0,1}; do
    for function in {0,1,2,3,4,5,6}; do
        for n_dims in {2,4,8,10,20,40,60,90,100,120,150}; do
            for n_execs in {1..30}; do
                python exec_times.py ${n_dims} ${function} ${parallel}>> exec_times.txt; 
                echo 'iteration '${n_execs}' ended';
            done
            echo 'Finished 30 execs for '${n_dims}' dims | function='${function};
        done
    done
    echo 'Finished half of tests!';
done

