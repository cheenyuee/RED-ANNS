num_servers=0
while IFS= read -r line; do
    num_servers=$((num_servers + 1))
done <hosts.mpi
echo "num_servers: $num_servers"

# para_path=app/deep10M_query10k_K4.json
para_path=app/deep100M_K4.json
K_VALUES=(10)
L_VALUES=(30 40 50 60 70 80 90 100 120 140 160)
T_VALUES=(4)
Sche_VALUES=(3)
Relax_VALUES=(0)
Cache_Nodes=(0)
for T in "${T_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        for L in "${L_VALUES[@]}"; do
            for Sche in "${Sche_VALUES[@]}"; do
                for Relax in "${Relax_VALUES[@]}"; do
                    for Cache in "${Cache_Nodes[@]}"; do
                        mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 numactl --cpunodebind=all --interleave=all ./build/tests/test_search_distributed config hosts ${para_path} $K $L $T $Sche $Relax $Cache
                        sleep 1
                    done
                done
            done
        done
    done
done