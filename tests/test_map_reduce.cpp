#include <iostream>
#include <unistd.h>
#include <thread>
#include <atomic>
#include "index.h"
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

class ResultMsg
{
public:
    double mean_hops;
    double mean_cmps;
    std::vector<std::vector<unsigned>> indices;
    std::vector<std::vector<float>> distances;
    ResultMsg() {}
    ResultMsg(double mean_hops, double mean_cmps, const std::vector<std::vector<unsigned>> &indices, const std::vector<std::vector<float>> &distances)
    {
        this->mean_hops = mean_hops;
        this->mean_cmps = mean_cmps;
        this->indices = indices;
        this->distances = distances;
    }
    template <class Archive>
    inline void serialize(Archive &ar, const unsigned int version)
    {
        ar & mean_hops;
        ar & mean_cmps;
        ar & indices;
        ar & distances;
    }
};

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        assert(false);
        exit(EXIT_FAILURE);
    }

    // set log level
    global_logger().set_log_level(LOG_EVERYTHING);
    // logstream(LOG_FATAL) << "This is LOG_FATAL " << LOG_endl;
    // logstream(LOG_ERROR) << "This is LOG_ERROR " << LOG_endl;
    // logstream(LOG_WARNING) << "This is LOG_WARNING " << LOG_endl;
    // logstream(LOG_EMPH) << "This is LOG_EMPH " << LOG_endl;
    // logstream(LOG_INFO) << "This is LOG_INFO " << LOG_endl;
    // logstream(LOG_DEBUG) << "This is LOG_DEBUG " << LOG_endl;

    // load global configs
    // load_config(string(argv[1]), world.size());

    // set the address file of host/cluster
    string host_fname = std::string(argv[2]);

    // MPI init
    boost::mpi::environment env(argc, argv, boost::mpi::threading::level::multiple);
    boost::mpi::communicator world;

    int sid = world.rank();        // 获取当前进程的排名
    int server_num = world.size(); // 获取总进程数
    std::string processor_name = env.processor_name();
    logstream(LOG_INFO) << "#" << sid << ": start process " << sid << " of " << server_num << ", running on " << processor_name << LOG_endl;
    ASSERT(Global::num_servers == server_num);

    // search
    string para_path = std::string(argv[3]);
    unsigned K = std::stoi(std::string(argv[4]));
    unsigned L = std::stoi(std::string(argv[5]));
    unsigned T = std::stoi(std::string(argv[6]));

    std::string jsonPath = para_path + std::to_string(sid) + std::string(".json");
    numaann::Parameters para;
    para.LoadConfigFromJSON(jsonPath);

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
    // index.load_coarse_clusters();
    index.generate_base_index();
    index.initialize_query_scratch(T, L);

    // processing......
    numaann::Parameters search_para;
    search_para.Set<unsigned>("L_search", L);

    std::vector<unsigned> bucket;
    // 用所有query测试
    for (size_t i = 0; i < query_num; i++)
    {
        bucket.push_back(i);
    }

    world.barrier();
    std::cout << "开始搜索..." << std::endl;
    std::atomic<unsigned> iter(0);
    size_t numThreads = T;
    std::vector<std::thread> threads;

    world.barrier();
    common::Timer timer;
    for (size_t threadId = 0; threadId < numThreads; ++threadId)
    {
        threads.push_back(std::thread([&, threadId]
                                      {
        std::cout << "threadId: " << threadId << std::endl;                        
        while (true) {
            unsigned tmp = iter.fetch_add(1);
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
    // world.barrier(); // 同步结束
    float seconds = timer.elapsed_seconds();
    std::cout << "#" << sid << " search finished." << std::endl;
    std::cout << "#" << sid << " search time(s): " << seconds << std::endl;
    std::cout << "#" << sid << " qps: " << ((float)bucket.size()) / seconds << std::endl;
    world.barrier();

    // transform local id to glabal id
    uint64_t prev_num = sid * index._base_num;
    for (unsigned i = 0; i < query_num; i++)
    {
        for (int k = 0; k < K; k++)
        {
            res_indices[i][k] += prev_num;
        }
    }

    // compute recall on every subgraph, then the total recall.
    float recall = common::compute_recall(query_num, K, gt, res_indices);
    printf("Recall@%d: %.2lf\n", K, recall);
    common::print_query_stats(query_num, stats);
    world.barrier();

    double mean_hops = common::get_mean_stats<unsigned>(stats, query_num,
                                                        [](const common::QueryStats &stats)
                                                        { return stats.n_hops; });
    double mean_cmps = common::get_mean_stats<unsigned>(stats, query_num,
                                                        [](const common::QueryStats &stats)
                                                        { return stats.n_cmps; });

    // collect from all servers
    std::vector<std::vector<unsigned>> indices_collected(query_num);
    std::vector<std::vector<float>> distances_collected(query_num);

    // gather results and compute the qps
    world.barrier();
    if (sid == 0)
    {
        indices_collected = res_indices;
        distances_collected = res_distances;
        for (size_t ssid = 1; ssid < server_num; ssid++)
        {
            ResultMsg msg;
            boost::mpi::request reqs = world.irecv(ssid, boost::mpi::any_tag, msg);
            // boost::mpi::request reqs = world.irecv(boost::mpi::any_source, boost::mpi::any_tag, msg);
            reqs.wait();
            mean_hops += msg.mean_hops;
            mean_cmps += msg.mean_cmps;
            for (size_t i = 0; i < query_num; i++)
            {
                indices_collected[i].insert(indices_collected[i].end(), msg.indices[i].begin(), msg.indices[i].end());
                distances_collected[i].insert(distances_collected[i].end(), msg.distances[i].begin(), msg.distances[i].end());
            }
        }
        logstream(LOG_EMPH) << "mean_hops: " << mean_hops << LOG_endl;
        logstream(LOG_EMPH) << "mean_cmps: " << mean_cmps << LOG_endl;
    }
    else
    {
        ResultMsg msg(mean_hops, mean_cmps, res_indices, res_distances);
        boost::mpi::request reqs = world.isend(0, sid, msg);
        reqs.wait(); // must wait here!
    }

    if (sid == 0)
    {
        seconds = timer.elapsed_seconds();
        logstream(LOG_EMPH) << "final qps: " << ((float)bucket.size()) / seconds << LOG_endl;
        // filter out the final results
        for (unsigned i = 0; i < query_num; i++)
        {
            ASSERT(indices_collected[i].size() == server_num * K);
            ASSERT(distances_collected[i].size() == server_num * K);

            std::vector<diskann::Neighbor> neighbors;
            for (size_t j = 0; j < server_num * K; j++)
            {
                neighbors.push_back(diskann::Neighbor(indices_collected[i][j], distances_collected[i][j]));
            }

            std::sort(neighbors.begin(), neighbors.end());
            for (size_t j = 0; j < server_num * K; j++)
            {
                indices_collected[i][j] = neighbors[j].id;
                distances_collected[i][j] = neighbors[j].distance;
            }
        }
        float recall = common::compute_recall(query_num, K, gt, indices_collected);
        logger(LOG_EMPH, "final Recall@%d: %.2lf\n", K, recall);
        logstream(LOG_EMPH) << "Baseline run para_path: " << para_path << LOG_endl;
        logstream(LOG_EMPH) << "K: " << K << LOG_endl;
        logstream(LOG_EMPH) << "L: " << L << LOG_endl;
        logstream(LOG_EMPH) << "T: " << T << LOG_endl;
    }

    // MPI exit
    logstream(LOG_INFO) << "#" << sid << ": Wait to exit." << LOG_endl;
    world.barrier();
    logstream(LOG_INFO) << "#" << sid << ": exit." << LOG_endl;
    return 0;
}