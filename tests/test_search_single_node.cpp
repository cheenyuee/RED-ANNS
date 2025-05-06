// numactl --cpunodebind=1 --membind=1 ./build/tests/test_search_single_node
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

    std::vector<std::vector<unsigned>> res_indices(query_num);
    for (unsigned i = 0; i < query_num; i++)
        res_indices[i].resize(K);

    std::vector<std::vector<float>> res_distances(query_num);
    for (unsigned i = 0; i < query_num; i++)
        res_distances[i].resize(K);

    common::QueryStats *stats = new common::QueryStats[query_num];

    numaann::Index index(para);
    index.load_base_data();
    index.load_base_graph();
    index.load_coarse_clusters();
    index.generate_base_index();
    index.initialize_query_scratch(T, L);

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
                    index.search_base_index(threadId, query_data + (uint64_t)i * query_dim, K, search_para, res_indices[i].data(), res_distances[i].data(), stats + i);
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
    float recall = common::compute_recall(query_num, K, gt, res_indices);
    printf("Recall@%d: %.2lf\n", K, recall);
    common::print_query_stats(query_num, stats);
    return 0;
}
