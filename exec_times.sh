for function in {0,1,2,3,4,5,6,7,8,9}; do
    for n_swarms in {1,2,5,10,15}; do
        for n_dims in {2,4,8,16,32,64,128}; do
            for n_execs in {1..10}; do
                python exec_times.py ${n_dims} ${n_swarms} ${function} >> exec_times.txt; 
                echo "iteration "${n_execs}" ended";
            done
            echo "Finished 10 execs for "${n_swarms}" n_swarms and "${n_dims}" dims | function="${function};
        done
    done
done

