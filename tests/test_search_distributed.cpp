#include <iostream>
#include <thread>
#include <atomic>
#include <boost/mpi.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/serialization/serialization.hpp>
#include "index.h"

class ResultMsg
{
public:
    float qps;
    float recall;
    unsigned query_count;
    double mean_hops;
    double mean_cmps;
    double mean_cmps_local;
    double mean_cmps_remote;
    double mean_latency;
    ResultMsg() {}
    ResultMsg(float qps, float recall) : qps(qps), recall(recall) {}
    ResultMsg(float qps, float recall, unsigned query_count, double mean_hops, double mean_cmps, double mean_cmps_local, double mean_cmps_remote, double mean_latency)
        : qps(qps), recall(recall), query_count(query_count), mean_hops(mean_hops), mean_cmps(mean_cmps), mean_cmps_local(mean_cmps_local), mean_cmps_remote(mean_cmps_remote), mean_latency(mean_latency) {}
    template <class Archive>
    inline void serialize(Archive &ar, const unsigned int version)
    {
        ar & qps;
        ar & recall;
        ar & query_count;
        ar & mean_hops;
        ar & mean_cmps;
        ar & mean_cmps_local;
        ar & mean_cmps_remote;
        ar & mean_latency;
    }
};

void test_search_distributed(boost::mpi::communicator &world, int sid, Mem *mem)
{
    // loading......
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
    index.load_base_index_distributed(sid, mem);
    index.load_coarse_clusters();
    index.initialize_query_scratch_distributed(Global::num_servers, T, L);
    world.barrier(); // 等待所有机器的数据都加载完成

    // N个server存储数据，1个server搜索
    // if (sid != 0)
    //     return;

    // process......
    numaann::Parameters search_para;
    search_para.Set<unsigned>("L_search", L);

    std::vector<unsigned> bucket;
    // 用所有query测试
    // for (size_t i = 0; i < query_num; i++)
    // {
    //     bucket.push_back(i);
    // }
    // 用指定query测试
    for (size_t i = 0; i < query_num; i++)
    {
        unsigned bucket_id = index.compute_closest_coarse_cluster(query_data + (uint64_t)i * query_dim);
        if (bucket_id == sid)
            bucket.push_back(i);
    }
    logstream(LOG_EMPH) << "#" << sid << ": bucket size = " << bucket.size() << LOG_endl;

    // index.set_cache(query_data, query_num, bucket, T, 10000000);

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
            // bind_to_core(threadId); // MPI中需要手动绑定thread到指定的core,线程编号不要超过core的数量(可能导致NUMA效应)
            while (true) {
                // size_t i = iter.fetch_add(1);
                // if(i < query_num) {
                //     index.search_......
                // } else {
                //     break;
                // }
                size_t tmp = iter.fetch_add(1);
                if(tmp < bucket.size()) {
                    size_t i = bucket[tmp];
                    // index.search_base_index_distributed(threadId, query_data + (uint64_t)i * query_dim, K, search_para, res[i].data(), 0);
                    index.search_base_index_distributed(threadId, query_data + (uint64_t)i * query_dim, K, search_para, res[i].data(), 0, 0, 3);
                } else {
                    break;
                }
            } }));
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    // world.barrier(); // 同步结束
    float seconds = timer.elapsed_seconds();
    std::cout << "#" << sid << " search finished." << std::endl;
    std::cout << "#" << sid << " search time(s): " << seconds << std::endl;
    // std::cout << "qps: " << ((float)query_num) / seconds << std::endl;
    float qps = ((float)bucket.size()) / seconds;
    std::cout << "#" << sid << " qps: " << qps << std::endl;
    float recall = common::compute_recall(query_num, K, gt, res);
    printf("Recall@%d: %.2lf\n", K, recall);
    index._gstore->print_io();

    // gather results and compute the qps
    world.barrier();
    if (sid == 0)
    {
        uint64_t total_qps = qps;
        float total_recall = recall;
        boost::mpi::request reqs;
        for (size_t src_sid = 1; src_sid < Global::num_servers; src_sid++)
        {
            ResultMsg msg;
            reqs = world.irecv(src_sid, boost::mpi::any_tag, msg);
            reqs.wait();
            total_qps += msg.qps;
            total_recall += msg.recall;
        }
        logstream(LOG_EMPH) << "final qps: " << total_qps << LOG_endl;
        logstream(LOG_EMPH) << "final recall: " << total_recall << LOG_endl;
    }
    else
    {
        boost::mpi::request reqs;
        ResultMsg msg(qps, recall);
        reqs = world.isend(0, sid, msg);
        reqs.wait(); // must wait here!
    }
}

enum TagValue
{
    proxy,
    engine
};

enum MsgType
{
    stop,
    enqueue,
    reorder,
    dequeue_high_affinity,
    dequeue_low_affinity,
    dequeue_succ,
    dequeue_fail
};

class QueryMsg
{
public:
    MsgType msg_type;
    unsigned query_id;
    unsigned learn_index_res;

    unsigned should_run_on_node;
    float data_affinity;

    QueryMsg() = default;
    QueryMsg(MsgType type) : msg_type(type) {}
    QueryMsg(MsgType type, unsigned id, unsigned res) : msg_type(type), query_id(id), learn_index_res(res) {}
    QueryMsg(MsgType type, unsigned id, unsigned res, unsigned node, float affinity) : msg_type(type), query_id(id), learn_index_res(res), should_run_on_node(node), data_affinity(affinity) {}
    template <class Archive>
    inline void serialize(Archive &ar, const unsigned int version)
    {
        ar & msg_type;
        ar & query_id;
        ar & learn_index_res;
        ar & should_run_on_node;
        ar & data_affinity;
    }
};

inline uint64_t rdma_fetch_add(int sid, int tid, Mem *mem)
{
    // 指定一个位置存放分布式共享变量
    int master = 0;
    uint64_t off = mem->ring_offset(0, 0);

    char *buf = mem->buffer(tid);

    RDMA &rdma = RDMA::get_rdma();
    rdma.dev->RdmaFetchAndAdd(tid, master, buf, off, 1);
    // logstream(LOG_EMPH) << "#" << sid << ": RdmaFetchAndAdd: " << *(uint64_t *)buf << std::endl;
    return *(uint64_t *)buf;
}

void test_search_distributed_with_dynamic_scheduling(boost::mpi::communicator &world, int sid, Mem *mem, const std::string &para_path, unsigned K, unsigned L, unsigned T, unsigned sche_strategy, unsigned relax, unsigned cache_node)
{
    // loading......
    numaann::Parameters para;
    para.LoadConfigFromJSON(para_path);

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

    common::QueryStats *stats = new common::QueryStats[query_num];

    numaann::Index index(para);
    index.load_base_index_distributed(sid, mem);
    index.load_coarse_clusters();
    index.initialize_query_scratch_distributed(Global::num_servers, T, L);
    world.barrier(); // 等待所有机器的数据都加载完成

    index.load_learn_data();
    index.load_learn_graph();
    index.generate_learn_projection();

    index.pq.load();

    /*****************/
    // 统计数据亲和性
    // if (sid == 0)
    // {
    //     std::unique_ptr<unsigned[]> query_distribute_by_topk(new unsigned[query_num]);
    //     std::unique_ptr<float[]> data_affinity(new float[query_num]);
    //     for (unsigned qid = 0; qid < query_num; qid++)
    //     {
    //         unsigned learn_index_res = index.beam_search_learn_graph(query_data + qid * query_dim, 10);
    //         unsigned should_run_on_node = 0;
    //         if (sche_strategy == 1) // 1.Random scheduling
    //             should_run_on_node = qid % Global::num_servers;
    //         if (sche_strategy == 2) // 2.IVF scheduling
    //             should_run_on_node = index.search_affinity(query_data + qid * query_dim).first;
    //         if (sche_strategy == 3) // 3.Graph scheduling
    //             should_run_on_node = index.search_affinity(learn_index_res).first;

    //         // 统计这个 query 的 top-k 的分布
    //         unsigned local_count = 0;
    //         for (size_t j = 0; j < K; j++)
    //         {
    //             unsigned topk = gt[qid][j];
    //             unsigned topk_cluster_index = index._base_in_bucket[topk];
    //             if (topk_cluster_index == should_run_on_node)
    //                 local_count++;
    //         }
    //         if (local_count == K)
    //             query_distribute_by_topk[qid] = 1; // 情况一
    //         else if (local_count > 0 and local_count < K)
    //             query_distribute_by_topk[qid] = 2; // 情况二
    //         else if (local_count == 0)
    //             query_distribute_by_topk[qid] = 3; // 情况三

    //         data_affinity[qid] = (float)local_count / K;
    //     }
    //     std::vector<unsigned> queries_case_1, queries_case_2, queries_case_3;
    //     float avg_data_affinity = 0.0;
    //     for (size_t qid = 0; qid < query_num; qid++)
    //     {
    //         if (query_distribute_by_topk[qid] == 1)
    //             queries_case_1.push_back(qid);
    //         if (query_distribute_by_topk[qid] == 2)
    //             queries_case_2.push_back(qid);
    //         if (query_distribute_by_topk[qid] == 3)
    //             queries_case_3.push_back(qid);
    //         avg_data_affinity += data_affinity[qid];
    //     }
    //     avg_data_affinity /= query_num;
    //     std::cout << "—————————————————— 比较三种情况 ——————————————————" << std::endl;
    //     std::cout << "Top-K: " << K << std::endl;
    //     std::cout << "目标区域在本端区域(情况一): " << queries_case_1.size() << std::endl;
    //     std::cout << "目标区域在交界区域(情况二): " << queries_case_2.size() << std::endl;
    //     std::cout << "目标区域在远端区域(情况三): " << queries_case_3.size() << std::endl;
    //     std::cout << "平均数据亲和度: " << avg_data_affinity << std::endl;

    //     // 保存data_affinity数据
    //     // std::ofstream outFile("vector_data_deep_PBS.bin", std::ios::binary);
    //     // if (!outFile)
    //     // {
    //     //     std::cerr << "无法打开文件!" << std::endl;
    //     // }
    //     // else
    //     // {
    //     //     size_t size = query_num;
    //     //     outFile.write(reinterpret_cast<char *>(&size), sizeof(size));
    //     //     outFile.write(reinterpret_cast<char *>(data_affinity.get()), size * sizeof(float));
    //     //     outFile.close();
    //     //     std::cout << "数据已成功保存到文件!" << std::endl;
    //     // }
    // }

    /*****************/

    // 测试Graph scheduling的准确性
    // float recall_in_learn = 0;
    // std::vector<int> sche_stat(index.bucket_count, 0);
    // for (size_t qid = 0; qid < query_num; qid++)
    // {
    //     unsigned learn_index_res = index.beam_search_learn_graph(query_data + qid * query_dim, 10);
    //     if (learn_index_res == index.beam_search_learn_graph(query_data + qid * query_dim, 1000))
    //         recall_in_learn++;

    //     unsigned should_run_on_node = index.search_affinity(learn_index_res).first;
    //     sche_stat[should_run_on_node]++;
    // }
    // recall_in_learn /= query_num;
    // std::cout << "recall_in_learn: " << recall_in_learn << ", sche_stat: ";
    // for (size_t i = 0; i < index.bucket_count; i++)
    // {
    //     std::cout << sche_stat[i] << ", ";
    // }
    // std::cout << std::endl;

    // process......
    numaann::Parameters search_para;
    search_para.Set<unsigned>("L_search", L);

    // std::thread proxy_thread([&, index = 0]
    //                          {
    //     boost::lockfree::queue<QueryMsg> taskQueue;
    //     while(true)
    //     {
    //         QueryMsg msg;
    //         boost::mpi::status stat = world.recv(boost::mpi::any_source, TagValue::proxy, msg);
    //         if(msg.msg_type == MsgType::stop)
    //         {
    //             break;
    //         }
    //         if(msg.msg_type == MsgType::enqueue)
    //         {
    //             taskQueue.push(msg);
    //         }
    //         else if(msg.msg_type == MsgType::dequeue)
    //         {
    //             bool succ = taskQueue.pop(msg);
    //             if(succ)
    //                 msg.msg_type = MsgType::dequeue_succ;
    //             else
    //                 msg.msg_type = MsgType::dequeue_fail;
    //             world.send(stat.source(), TagValue::engine, msg);
    //         }
    //         else
    //         {
    //             throw std::runtime_error("error@MsgType in proxy.");
    //         }
    //     } });

    // std::thread proxy_thread([&, index = 0]
    //                          {
    //     // std::priority_queue<QueryMsg, std::vector<QueryMsg>, std::function<bool(const QueryMsg &, const QueryMsg &)>> taskQueue([](const QueryMsg &a, const QueryMsg &b)
    //     //                                                                                                                   {
    //     //     return a.data_affinity < b.data_affinity;
    //     // });
    //     std::vector<QueryMsg> taskQueue;
    //     bool reorder_done = false;

    //     while(true)
    //     {
    //         QueryMsg msg;
    //         boost::mpi::status stat = world.recv(boost::mpi::any_source, TagValue::proxy, msg);
    //         if(msg.msg_type == MsgType::stop)
    //         {
    //             break;
    //         }
    //         else if(msg.msg_type == MsgType::enqueue)
    //         {
    //             // taskQueue.push(msg);
    //             taskQueue.push_back(msg);
    //         }
    //         else if(msg.msg_type == MsgType::reorder and !reorder_done)
    //         {
    //             // 按 data_affinity 升序重排
    //             std::sort(taskQueue.begin(), taskQueue.end(), [](const QueryMsg &a, const QueryMsg &b)
    //                       {
    //                 return a.data_affinity < b.data_affinity;
    //             });
    //             reorder_done = true;
    //         }
    //         else if(msg.msg_type == MsgType::dequeue_high_affinity or msg.msg_type == MsgType::dequeue_low_affinity)
    //         {
    //             if(!taskQueue.empty())
    //             {
    //                 // msg = taskQueue.top();
    //                 // taskQueue.pop();
    //                 msg = taskQueue.back();
    //                 taskQueue.pop_back();
    //                 msg.msg_type = MsgType::dequeue_succ;
    //             }
    //             else
    //             {
    //                 msg.msg_type = MsgType::dequeue_fail;
    //             }
    //             world.send(stat.source(), TagValue::engine, msg);
    //         }
    //         else
    //         {
    //             throw std::runtime_error("error@MsgType in proxy.");
    //         }
    //     } });

    std::thread proxy_thread([&, index = 0]
                             {
        boost::circular_buffer<QueryMsg> taskQueue(query_num);
        bool reorder_done = false;

        while(true)
        {
            QueryMsg msg;
            boost::mpi::status stat = world.recv(boost::mpi::any_source, TagValue::proxy, msg);
            if(msg.msg_type == MsgType::stop)
            {
                break;
            }
            else if(msg.msg_type == MsgType::enqueue)
            {
                taskQueue.push_back(msg);
            }
            else if(msg.msg_type == MsgType::reorder)
            {
                if(!reorder_done)
                {
                    std::sort(taskQueue.begin(), taskQueue.end(), [](const QueryMsg &a, const QueryMsg &b)
                    {
                        return a.data_affinity < b.data_affinity;
                    });
                    reorder_done = true;
                }
            }
            else if(msg.msg_type == MsgType::dequeue_high_affinity)
            {
                if(!taskQueue.empty())
                {
                    msg = taskQueue.back();
                    taskQueue.pop_back();
                    msg.msg_type = MsgType::dequeue_succ;
                }
                else
                {
                    msg.msg_type = MsgType::dequeue_fail;
                }
                world.send(stat.source(), TagValue::engine, msg);
            }
            else if(msg.msg_type == MsgType::dequeue_low_affinity)
            {
                if(!taskQueue.empty())
                {
                    msg = taskQueue.front();
                    taskQueue.pop_front();
                    msg.msg_type = MsgType::dequeue_succ;
                }
                else
                {
                    msg.msg_type = MsgType::dequeue_fail;
                }
                world.send(stat.source(), TagValue::engine, msg);
            }
            else
            {
                throw std::runtime_error("error@MsgType in proxy.");
            }
        } });

    // world.barrier();
    // std::vector<unsigned> bucket;
    // std::string cache_test(para.Get<std::string>("cache_test"));
    // unsigned cache_test_num, cache_test_dim;
    // float *cache_test_data = common::read_data(cache_test, file_format, cache_test_num, cache_test_dim);
    // std::cout << "cache_test_num, cache_test_dim = " << cache_test_num << ", " << cache_test_dim << std::endl;
    // for (size_t i = 0; i < cache_test_num; i++)
    // {
    //     if (cache_node == 0)
    //         continue;
    //     if (sche_strategy != 3) // 3.Graph scheduling
    //         throw std::runtime_error("error@sche_strategy in caching.");

    //     unsigned learn_index_res = index.beam_search_learn_graph(cache_test_data + (uint64_t)i * cache_test_dim, 10);
    //     unsigned should_run_on_node = index.search_affinity(learn_index_res).first;
    //     if (should_run_on_node == sid)
    //         bucket.push_back(i);
    // }
    // logstream(LOG_EMPH) << "#" << sid << ": bucket size = " << bucket.size() << LOG_endl;
    // index.set_cache(cache_test_data, cache_test_num, bucket, T, cache_node);
    // delete[] cache_test_data;
    // index._gstore->rdma_cache->lookup_cnt = 0;
    // index._gstore->rdma_cache->hit_cnt = 0;
    // index._gstore->rdma_cache->total_ns = 0;

    world.barrier();
    std::atomic<unsigned> run_query_num(0);
    size_t numThreads = T;
    std::vector<std::thread> threads;

    /*****************/
    // 测每秒能进行多少次距离计算，分local和remote
    // if (sid == 0)
    // {
    //     uint64_t NUM_ITERATIONS = 2000000, batch_size = 64;
    //     std::cout << "#" << sid << " NUM_ITERATIONS: " << NUM_ITERATIONS << std::endl;
    //     std::cout << "#" << sid << " batch_size: " << batch_size << std::endl;
    //     for (size_t threadId = 0; threadId < numThreads; ++threadId)
    //     {
    //         threads.push_back(std::thread([&, threadId]
    //                                       {
    //                    std::cout << "threadId: " << threadId << std::endl;
    //                    common::Timer timer;
    //                    index.test_compute(threadId, query_data, NUM_ITERATIONS, batch_size);
    //                    float seconds = timer.elapsed_seconds();
    //                    std::cout << "#" << sid << " dist num per second: " << NUM_ITERATIONS * batch_size / seconds << std::endl;
    //                    std::cout << "#" << sid << " second per iter: " << seconds / NUM_ITERATIONS  << std::endl;
    //                 }));
    //     }
    //     for (auto &thread : threads)
    //     {
    //         thread.join();
    //     }
    //     threads.clear();
    // }
    // world.barrier();
    /*****************/

    world.barrier(); // 同步开始
    common::Timer timer;
    for (size_t threadId = 0; threadId < numThreads; ++threadId)
    {
        threads.push_back(std::thread([&, threadId]
                                      {
                                          std::cout << "threadId: " << threadId << std::endl;
                                          // bind_to_core(threadId); // MPI中需要手动绑定thread到指定的core,线程编号不要超过core的数量(可能导致NUMA效应)
                                          /* 任务分发 */
                                          while (true)
                                          {
                                              size_t qid = rdma_fetch_add(sid, threadId, mem);
                                              if (qid < query_num)
                                              {
                                                  // unsigned learn_index_res = index.search_learn_graph(query_data + i * query_dim);
                                                  unsigned learn_index_res = index.beam_search_learn_graph(query_data + qid * query_dim, 10);

                                                  if(sche_strategy == 1)
                                                  {
                                                    unsigned should_run_on_node = qid % Global::num_servers;
                                                    world.send(should_run_on_node, TagValue::proxy, QueryMsg(MsgType::enqueue, qid, learn_index_res));
                                                  }

                                                  if(sche_strategy == 2)
                                                  {
                                                    std::pair<unsigned, float> sche_info = index.search_affinity(query_data + qid * query_dim);
                                                    world.send(sche_info.first, TagValue::proxy, QueryMsg(MsgType::enqueue, qid, learn_index_res, sche_info.first, sche_info.second));
                                                  }

                                                  if(sche_strategy == 3)
                                                  {
                                                    std::pair<unsigned, float> sche_info = index.search_affinity(learn_index_res);
                                                    world.send(sche_info.first, TagValue::proxy, QueryMsg(MsgType::enqueue, qid, learn_index_res, sche_info.first, sche_info.second));
                                                  }   
                                              }
                                              else
                                              {
                                                  break;
                                              }
                                          }
                                          /* 任务重排 */
                                          // world.send(sid, TagValue::proxy, QueryMsg(MsgType::reorder));
                                          /* 任务执行（注意不要漏query) */
                                          // int nsvr_has_query = 1;
                                          int nsvr_has_query = Global::num_servers;
                                          std::vector<bool> has_query(Global::num_servers, true);
                                          size_t dsid = 0;
                                          while (true)
                                          {
                                              if(nsvr_has_query == 0)
                                                  break;

                                              if(has_query[sid])
                                              {
                                                dsid = sid;
                                                world.send(dsid, TagValue::proxy, QueryMsg(MsgType::dequeue_high_affinity));
                                              }
                                              else
                                              {
                                                dsid = (dsid + 1) % Global::num_servers;
                                                if(!has_query[dsid])
                                                    continue;
                                                world.send(dsid, TagValue::proxy, QueryMsg(MsgType::dequeue_low_affinity));
                                              }
                                              QueryMsg msg;
                                              boost::mpi::status stat = world.recv(dsid, TagValue::engine, msg);
                                              if(msg.msg_type == MsgType::dequeue_succ)
                                              {
                                                  size_t qid = msg.query_id;
                                                  unsigned learn_index_res = msg.learn_index_res;
                                                  // index.search_base_index(threadId, query_data + i * query_dim, K, search_para, res[i].data(), learn_index_res, 100);
                                                  index.search_base_index_distributed(threadId, query_data + (uint64_t)qid * query_dim, K, search_para, res[qid].data(), learn_index_res, 10, relax, stats + qid);
                                                  run_query_num++;
                                              } else if(msg.msg_type == MsgType::dequeue_fail) {
                                                  has_query[dsid] = false;
                                                  nsvr_has_query--;
                                              } else {
                                                  throw std::runtime_error("error@MsgType in engine.");
                                              }
                                          } }));
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    world.barrier(); // 同步结束
    float seconds = timer.elapsed_seconds();
    std::cout << "#" << sid << " search finished." << std::endl;
    std::cout << "#" << sid << " search time(s): " << seconds << std::endl;

    // double hit_rate = (double)index._gstore->rdma_cache->hit_cnt / index._gstore->rdma_cache->lookup_cnt;
    // std::cout << "#" << sid << " lookup_cnt: " << index._gstore->rdma_cache->lookup_cnt << ", hit_cnt: " << index._gstore->rdma_cache->hit_cnt << ", hit_rate: " << hit_rate * 100 << "%" << std::endl;
    // std::cout << "#" << sid << " total_ns: " << index._gstore->rdma_cache->total_ns << " single_ns: " << (double)index._gstore->rdma_cache->total_ns / index._gstore->rdma_cache->lookup_cnt << std::endl;

    unsigned query_count = run_query_num;
    float qps = ((float)query_count) / seconds;
    std::cout << "#" << sid << " query_count: " << query_count << " qps: " << qps << std::endl;

    float recall = common::compute_recall(query_num, K, gt, res);
    printf("Recall@%d: %.2lf\n", K, recall);

    double mean_hops = common::get_mean_stats<unsigned>(stats, query_num,
                                                        [](const common::QueryStats &stats)
                                                        { return stats.n_hops; });
    double mean_cmps = common::get_mean_stats<unsigned>(stats, query_num,
                                                        [](const common::QueryStats &stats)
                                                        { return stats.n_cmps; });
    double mean_cmps_local = common::get_mean_stats<unsigned>(stats, query_num,
                                                              [](const common::QueryStats &stats)
                                                              { return stats.n_cmps_local; });
    double mean_cmps_remote = common::get_mean_stats<unsigned>(stats, query_num,
                                                               [](const common::QueryStats &stats)
                                                               { return stats.n_cmps_remote; });
    // mean_cmps_remote *= (1 - hit_rate);
    double mean_latency = common::get_mean_stats<float>(stats, query_num,
                                                        [](const common::QueryStats &stats)
                                                        { return stats.total_us; });

    // gather results and compute the qps
    world.barrier();
    if (sid == 0)
    {
        boost::mpi::request reqs;
        for (size_t src_sid = 1; src_sid < Global::num_servers; src_sid++)
        {
            ResultMsg msg;
            reqs = world.irecv(src_sid, boost::mpi::any_tag, msg);
            reqs.wait();
            qps += msg.qps;
            recall += msg.recall;
            query_count += msg.query_count;
            mean_hops += msg.mean_hops;
            mean_cmps += msg.mean_cmps;
            mean_cmps_local += msg.mean_cmps_local;
            mean_cmps_remote += msg.mean_cmps_remote;
            mean_latency += msg.mean_latency;
        }
        logstream(LOG_EMPH) << "DSM-ANNS run para_path: " << para_path << LOG_endl;
        logstream(LOG_EMPH) << "K: " << K << LOG_endl;
        logstream(LOG_EMPH) << "L: " << L << LOG_endl;
        logstream(LOG_EMPH) << "T: " << T << LOG_endl;
        logstream(LOG_EMPH) << "sche_strategy: " << sche_strategy << LOG_endl;
        logstream(LOG_EMPH) << "relax: " << relax << LOG_endl;

        logstream(LOG_EMPH) << std::fixed << std::setprecision(2);
        logstream(LOG_EMPH) << "qps: " << qps << LOG_endl;
        logstream(LOG_EMPH) << "recall: " << recall << LOG_endl;
        logstream(LOG_EMPH) << "query_count: " << query_count << LOG_endl;
        logstream(LOG_EMPH) << "mean_hops: " << mean_hops << LOG_endl;
        logstream(LOG_EMPH) << "mean_cmps: " << mean_cmps << LOG_endl;
        logstream(LOG_EMPH) << "mean_cmps_local: " << mean_cmps_local << LOG_endl;
        logstream(LOG_EMPH) << "mean_cmps_remote: " << mean_cmps_remote << LOG_endl;
        logstream(LOG_EMPH) << "mean_latency: " << mean_latency << LOG_endl;
    }
    else
    {
        boost::mpi::request reqs;
        ResultMsg msg(qps, recall, query_count, mean_hops, mean_cmps, mean_cmps_local, mean_cmps_remote, mean_latency);
        reqs = world.isend(0, sid, msg);
        reqs.wait(); // must wait here!
    }

    // exit proxy_thread
    world.barrier();
    world.send(sid, TagValue::proxy, QueryMsg(MsgType::stop));
    proxy_thread.join();
}

int main(int argc, char *argv[])
{
    if (argc != 10)
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
    boost::mpi::environment env(argc, argv, boost::mpi::threading::level::multiple); // 这里设为multiple
    boost::mpi::communicator world;

    int sid = world.rank();        // 获取当前进程的排名
    int server_num = world.size(); // 获取总进程数
    std::string processor_name = env.processor_name();
    logstream(LOG_INFO) << "#" << sid << ": start process " << sid << " of " << server_num << ", running on " << processor_name << LOG_endl;
    assert(Global::num_servers == server_num);
    ASSERT(Global::num_servers == server_num);

    // load CPU topology by hwloc
    load_node_topo();
    logstream(LOG_INFO) << "#" << sid << ": has " << num_cores << " cores." << LOG_endl;

    // allocate memory regions
    std::vector<RDMA::MemoryRegion> mrs;
    Mem *mem = new Mem(Global::num_servers, Global::num_threads);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(mem->size()) << "GB memory" << LOG_endl;
    RDMA::MemoryRegion mr_cpu = {RDMA::MemType::CPU, mem->address(), mem->size(), mem};
    mrs.push_back(mr_cpu);

    // RDMA_init
    RDMA_init(Global::num_servers, Global::num_threads, sid, mrs, host_fname);

    world.barrier();

    // USE data......
    // test_search_distributed(world, sid, mem);
    string para_path = std::string(argv[3]);
    unsigned K = std::stoi(std::string(argv[4]));
    unsigned L = std::stoi(std::string(argv[5]));
    unsigned T = std::stoi(std::string(argv[6]));
    unsigned sche_strategy = std::stoi(std::string(argv[7]));
    unsigned relax = std::stoi(std::string(argv[8]));
    unsigned cache_node = std::stoi(std::string(argv[9]));
    test_search_distributed_with_dynamic_scheduling(world, sid, mem, para_path, K, L, T, sche_strategy, relax, cache_node);

    // MPI exit
    logstream(LOG_INFO) << "#" << sid << ": Wait to exit." << LOG_endl;
    world.barrier();
    logstream(LOG_INFO) << "#" << sid << ": exit." << LOG_endl;
    return 0;
}
