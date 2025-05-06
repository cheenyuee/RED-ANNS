#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <boost/mpi.hpp>
#include <algorithm>
#include "mem.hpp"
#include "bind.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        assert(false);
        exit(EXIT_FAILURE);
    }

    // load global configs
    // load_config(string(argv[1]), world.size());

    // set the address file of host/cluster
    string host_fname = std::string(argv[2]);

    // MPI init
    boost::mpi::environment env(argc, argv, boost::mpi::threading::level::funneled);
    boost::mpi::communicator world;

    int server_id = world.rank();  // 获取当前进程的排名
    int server_num = world.size(); // 获取总进程数
    std::string processor_name = env.processor_name();
    logstream(LOG_INFO) << "#" << server_id << ": start process " << server_id << " of " << server_num << ", running on " << processor_name << LOG_endl;
    assert(Global::num_servers == server_num);
    ASSERT(Global::num_servers == server_num);

    // set log level
    global_logger().set_log_level(LOG_EVERYTHING);

    // allocate memory regions
    vector<RDMA::MemoryRegion> mrs;
    Mem *mem = new Mem(Global::num_servers, Global::num_threads);
    std::cout << "#" << server_id << ": allocate " << B2GiB(mem->size()) << "GB memory" << std::endl;
    RDMA::MemoryRegion mr_cpu = {RDMA::MemType::CPU, mem->address(), mem->size(), mem};
    mrs.push_back(mr_cpu);

    // RDMA_init
    RDMA_init(Global::num_servers, Global::num_threads, server_id, mrs, host_fname);

    // initiate local buffer
    int dst_sid = server_id;
    for (int tid = 0; tid < Global::num_threads; tid++)
    {
        char *buf = mem->buffer(tid);
        strcpy(buf, "Proof that u read something.");
        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaWrite(tid, dst_sid, buf, 29, 0);
    }
    world.barrier();

    /* 测试 bandwidth，使用mlnx_perf监测流量 */
    // if (server_id == 0)
    // {
    //     std::vector<std::thread> threads;
    //     for (int threadId = 0; threadId < Global::num_threads; threadId++)
    //     {
    //         threads.push_back(std::thread([&, threadId]
    //                                       {
    //         bind_to_core(threadId);
    //         // len=16000，fixed read，T1：11.77 GBps（快要跑满100Gbps了）
    //         int tid = threadId;
    //         char *buf = mem->buffer(tid);
    //         uint64_t len = 1000;
    //         uint64_t off = 0;
    //         RDMA &rdma = RDMA::get_rdma();
    //         // 为什么 TestRdmaBW 时 mlnx_perf 监测的流量数据对不上
    //         rdma.dev->TestRdmaBW(tid, dst_sid, buf, len, off, (uint64_t)10 * 1000 * 1000, 128); }));
    //     }
    //     for (auto &thread : threads)
    //     {
    //         thread.join();
    //     }
    // }
    // world.barrier();
    // return 0;

    // test_rdma_lat
    uint64_t len = (uint64_t)1024;
    uint64_t msg_num = GiB2B(Global::memstore_size_gb) / len;
    uint64_t NUM_ITERATIONS = (uint64_t)1000000;
    if (server_id == 0)
        dst_sid = 1;
    if (server_id == 1)
        dst_sid = 0;

    if (server_id == 0)
    {
        logger(LOG_EMPH, "#%d: Test 1 thread RdmaReadBatch(parallel)...", server_id);
        double total_time = 0; // total_time的单位是微秒
        size_t batch_size = 32;
        // size_t batch_size = 1;
        for (uint64_t i = 0; i < NUM_ITERATIONS; i++)
        {
            std::vector<int> nids;
            std::vector<char *> local_batch;
            std::vector<uint64_t> off_batch;

            int tid = 0;
            char *buf = mem->buffer(tid);
            RDMA &rdma = RDMA::get_rdma();
            for (size_t j = 0; j < batch_size; j++)
            {
                // uint64_t off = ((i + j) % msg_num) * len; // sequential read
                uint64_t off = (rand() % msg_num) * len; // random read
                nids.push_back(1);
                local_batch.push_back(buf);
                off_batch.push_back(off);
                buf += len;
            }
            auto start = std::chrono::high_resolution_clock::now();
            rdma.dev->RdmaReadBatch(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size);

            // std::vector<int> polls(Global::num_servers, 0);
            // rdma.dev->RdmaReadBatch_Async_Send(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size, polls);
            // rdma.dev->RdmaReadBatch_Async_Wait(tid, polls);

            // rdma.dev->RdmaReadBatchDoorbell(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size);

            // std::vector<int> polls(Global::num_servers, 0);
            // auto time0 = std::chrono::high_resolution_clock::now();
            // rdma.dev->RdmaReadBatchDoorbell_Async_Send(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size, polls);
            // while (true)
            // {
            //     auto ckpt1 = std::chrono::high_resolution_clock::now();
            //     if (std::chrono::duration_cast<std::chrono::nanoseconds>(ckpt1 - time0).count() / 1e3 > 20)
            //         break;
            // }
            // auto time1 = std::chrono::high_resolution_clock::now();
            // rdma.dev->RdmaReadBatch_Async_Wait(tid, polls);
            // auto time2 = std::chrono::high_resolution_clock::now();
            // auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();
            // auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
            // std::cout << "batch_size: " << batch_size << ", post duration: " << (double)duration1 / 1000 << ", poll duration: " << (double)duration2 / 1000 << ", total duration: " << (double)(duration1 + duration2) / 1000 << " us" << std::endl;

            /* use unsignal */
            // rdma.dev->RdmaReadBatchUnsignal(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size);

            // std::vector<int> polls(Global::num_servers, 0);
            // rdma.dev->RdmaReadBatchUnsignal_Async_Send(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size, polls);
            // rdma.dev->RdmaReadBatchUnsignal_Async_Wait(tid, polls);

            // std::vector<int> polls(Global::num_servers, 0);
            // auto time0 = std::chrono::high_resolution_clock::now();
            // rdma.dev->RdmaReadBatchDoorbellUnsignal_Async_Send(tid, nids.data(), local_batch.data(), len, off_batch.data(), batch_size, polls);
            // while (true)
            // {
            //     auto ckpt1 = std::chrono::high_resolution_clock::now();
            //     if (std::chrono::duration_cast<std::chrono::nanoseconds>(ckpt1 - time0).count() / 1e3 > 20)
            //         break;
            // }
            // auto time1 = std::chrono::high_resolution_clock::now();
            // rdma.dev->RdmaReadBatchUnsignal_Async_Wait(tid, polls);
            // auto time2 = std::chrono::high_resolution_clock::now();
            // auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();
            // auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
            // std::cout << "batch_size: " << batch_size << ", post duration: " << (double)duration1 / 1000 << ", poll duration: " << (double)duration2 / 1000 << ", total duration: " << (double)(duration1 + duration2) / 1000 << " us" << std::endl;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_time += (double)duration / 1e3;
        };
        logger(LOG_INFO, "#%d: READ batch size %ld latency: %lf us", server_id, batch_size, (double)total_time / NUM_ITERATIONS);
        logger(LOG_INFO, "#%d: READ batch size %ld bytes bandwidth: %lf GBps", server_id, batch_size, B2GiB(1e6 * len * batch_size * NUM_ITERATIONS / (double)total_time));
    }
    world.barrier();
    // return 0;

    /* test 1 thread RdmaRead */
    logger(LOG_EMPH, "#%d: Test 1 thread READ...", server_id);
    double total_time = 0; // total_time的单位是微秒
    for (uint64_t i = 0; i < NUM_ITERATIONS; i++)
    {
        int tid = 0;
        char *buf = mem->buffer(tid);
        RDMA &rdma = RDMA::get_rdma();
        // uint64_t off = (i % msg_num) * len; // sequential read
        uint64_t off = (rand() % msg_num) * len; // random read
        auto start = std::chrono::high_resolution_clock::now();
        rdma.dev->RdmaRead(tid, dst_sid, buf, len, off);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        total_time += (double)duration / 1e3;
    };
    logger(LOG_INFO, "#%d: READ %ld bytes latency: %lf us", server_id, len, (double)total_time / NUM_ITERATIONS);
    logger(LOG_INFO, "#%d: READ %ld bytes bandwidth: %lf GBps", server_id, len, B2GiB(1e6 * len * NUM_ITERATIONS / (double)total_time));
    world.barrier();

    /* test concurrent RdmaRead */
    logger(LOG_EMPH, "#%d: Test concurrent READ...", server_id);
    double total_time_per_thd[Global::num_threads]{0};
    std::vector<std::thread> threads;
    for (int threadId = 0; threadId < Global::num_threads; threadId++)
    {
        threads.push_back(std::thread([&, threadId]
                                      {
            bind_to_core(threadId);
            double total_time = 0; // total_time的单位是微秒
            for (uint64_t i = 0; i < NUM_ITERATIONS; i++)
            {
                int tid = threadId;
                char *buf = mem->buffer(tid);
                RDMA &rdma = RDMA::get_rdma();
                // uint64_t off = (i % msg_num) * len; // sequential read
                uint64_t off = (rand() % msg_num) * len; // random read
                auto start = std::chrono::high_resolution_clock::now();
                rdma.dev->RdmaRead(tid, dst_sid, buf, len, off);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                total_time += (double)duration / 1e3;
            };
            total_time_per_thd[threadId] = total_time; }));
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    double sum_total_time = std::accumulate(total_time_per_thd, total_time_per_thd + Global::num_threads, 0);
    double avg_total_time = (double)sum_total_time / Global::num_threads;
    logger(LOG_INFO, "#%d: concurrent READ %ld bytes latency: %lf us", server_id, len, (double)avg_total_time / NUM_ITERATIONS);
    logger(LOG_INFO, "#%d: concurrent READ %ld bytes bandwidth: %lf GBps", server_id, len, B2GiB(1e6 * Global::num_threads * len * NUM_ITERATIONS / (double)avg_total_time));
    world.barrier();

    // MPI exit
    logstream(LOG_INFO) << "#" << server_id << ": Wait to exit." << LOG_endl;
    world.barrier();
    logstream(LOG_INFO) << "#" << server_id << ": exit." << LOG_endl;
    return 0;
}