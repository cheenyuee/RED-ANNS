add_executable(test_rdma_lat test_rdma_lat.cpp)
target_link_libraries(test_rdma_lat -libverbs -pthread -lboost_mpi -lhwloc)

add_executable(test_search_distributed test_search_distributed.cpp)
target_link_libraries(test_search_distributed ${PROJECT_NAME} -libverbs -pthread -lboost_mpi -lboost_serialization -lhwloc)

add_executable(test_search_membkt test_search_membkt.cpp)
target_link_libraries(test_search_membkt ${PROJECT_NAME} -libverbs -pthread -lboost_mpi -lhwloc)

add_executable(test_search_single_node test_search_single_node.cpp)
target_link_libraries(test_search_single_node ${PROJECT_NAME} -libverbs -pthread -lboost_mpi -lhwloc)

add_executable(test_map_reduce test_map_reduce.cpp)
target_link_libraries(test_map_reduce ${PROJECT_NAME} -libverbs -pthread -lboost_mpi -lboost_serialization -lhwloc)

add_executable(test_pq test_pq.cpp)
target_link_libraries(test_pq ${PROJECT_NAME} -pthread)