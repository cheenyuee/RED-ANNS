# 一行即一个server
num_servers=0
while IFS= read -r line; do
    num_servers=$((num_servers + 1))
done <hosts.mpi
echo "num_servers: $num_servers"

# mpiexec -hostfile hosts.mpi -n $num_servers ./build/tests/test_search_distributed config hosts

# 指定使用网络接口 eno1
# mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 ./build/tests/test_search_distributed config hosts
# mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 ./build/tests/test_rdma_lat config hosts

# 单 NUMA Node
mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 numactl --cpunodebind=1 --membind=1 ./build/tests/test_search_distributed config hosts app/deep10M_query10k_K4.json 10 100 1
# 多 NUMA Node
# mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 numactl --cpunodebind=all --interleave=all ./build/tests/test_search_distributed config hosts

# 测试test_map_reduce
# mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 numactl --cpunodebind=1 --membind=1 ./build/tests/test_map_reduce config hosts


# 定义 K、L 和 T 的参数范围
para_path=app/deep10M_query10k_K4.json
K_VALUES=(10)
L_VALUES=(100)
T_VALUES=(1)
Sche_VALUES=(1 2 3)

# 遍历 K、L 和 T 的所有组合
for K in "${K_VALUES[@]}"; do
    for L in "${L_VALUES[@]}"; do
        for T in "${T_VALUES[@]}"; do
            for Sche in "${Sche_VALUES[@]}"; do
                echo "para_path=${para_path} Running with K=$K, L=$L, T=$T, Sche=$Sche"
                mpiexec -hostfile hosts.mpi -n $num_servers --mca btl_tcp_if_include eno1 numactl --cpunodebind=1 --membind=1 ./build/tests/test_search_distributed config hosts ${para_path} $K $L $T $Sche
                sleep 1
            done
        done
    done
done