// numactl --cpunodebind=1 --membind=1 ./build/tests/test_search_membkt
#include <iostream>
#include <thread>
#include <atomic>
#include "index.h"

int main(int argc, char *argv[])
{
    global_logger().set_log_level(LOG_EVERYTHING);

    // load......
    numaann::Parameters para;
    para.LoadConfigFromJSON("./app/deep10M_query10k_K4.json");

    unsigned K(para.Get<unsigned>("K"));
    unsigned L(para.Get<unsigned>("L"));
    unsigned T(para.Get<unsigned>("T"));
    if (L < K)
        throw std::runtime_error("error: search_L cannot be smaller than search_K.");
    if (T > Global::num_threads)
        throw std::runtime_error("error: search_T cannot be bigger than Global::num_threads.");

    std::string file_format(para.Get<std::string>("file_format"));
    std::string query_file(para.Get<std::string>("query_file"));
    std::string gt_file(para.Get<std::string>("gt_file"));

    unsigned query_num, query_dim;
    float *query_data = common::read_data(query_file, file_format, query_num, query_dim);
    std::cout << "query_num, query_dim = " << query_num << ", " << query_dim << std::endl;

    unsigned gt_num, gt_dim;
    std::vector<std::vector<unsigned>> gt = common::read_gt(gt_file, file_format, gt_num, gt_dim);
    std::cout << "gt_num, gt_dim = " << gt_num << ", " << gt_dim << std::endl;

    std::vector<std::vector<unsigned>> res(query_num);
    for (unsigned i = 0; i < query_num; i++)
        res[i].resize(K);

    numaann::Index index(para);
    index.load_base_data();
    index.load_base_graph();
    index.load_coarse_clusters();
    index.generate_base_index_on_buckets();
    index.save_base_index_on_buckets();
    index.initialize_query_scratch_distributed(index.bucket_count, T, L);

    /* 原始排列 */
    // size_t data_size;
    // index._memeory_buckets[0] = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_0", data_size);
    // index._memeory_buckets[1] = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1", data_size);
    // index._membkt_EP = std::make_pair((server_id_t)1, (local_id_t)3286336);

    // process......
    numaann::Parameters search_para;
    search_para.Set<unsigned>("L_search", L);

    std::vector<unsigned> bucket;
    // 用所有query测试
    for (size_t i = 0; i < query_num; i++)
    {
        bucket.push_back(i);
    }
    // 用指定query测试
    // for (size_t i = 0; i < query_num; i++)
    // {
    //     unsigned bucket_id = index.compute_closest_coarse_cluster(query_data + (uint64_t)i * query_dim);
    //     if (bucket_id == 0)
    //         bucket.push_back(i);
    // }

    std::cout << "开始搜索..." << std::endl;
    std::atomic<unsigned> iter(0);
    size_t numThreads = T;
    std::vector<std::thread> threads;
    common::Timer timer;
    for (size_t threadId = 0; threadId < numThreads; ++threadId)
    {
        threads.push_back(std::thread([&, threadId]
                                      {
            std::cout << "threadId: " << threadId << std::endl;
            while (true) {
                size_t tmp = iter.fetch_add(1);
                if(tmp < bucket.size()) {
                    size_t i = bucket[tmp];
                    // search_base_index_on_buckets 比 search 的qps低5%，可能是数据排布的原因？可能是使用二维ID的原因？
                    index.search_base_index_on_buckets(threadId, query_data + (uint64_t)i * query_dim, K, search_para, res[i].data());
                } else {
                    break;
                }
            } }));
    }
    std::cout << "等待执行完成..." << std::endl;
    for (auto &thread : threads)
    {
        thread.join();
    }
    float seconds = timer.elapsed_seconds();
    std::cout << "search finished." << std::endl;
    std::cout << "search time(s): " << seconds << std::endl;
    // std::cout << "qps: " << ((float)query_num) / seconds << std::endl;
    std::cout << "qps: " << ((float)bucket.size()) / seconds << std::endl;
    float recall = common::compute_recall(query_num, K, gt, res);
    printf("Recall@%d: %.2lf\n", K, recall);
    return 0;
}
