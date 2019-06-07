for function in {0,1,2,3,4,5,6}; do
    for n_swarms in {5,10,15}; do
        for n_dims in {2,4,8,10,20,90,100,150}; do
            for n_execs in {1..20}; do
                python exec_times.py ${n_dims} ${n_swarms} ${function} >> exec_times.txt; 
                echo "iteration "${n_execs}" ended";
            done
            echo "Finished 20 execs for "${n_dims}" dims | function="${function};
        done
    done
done

