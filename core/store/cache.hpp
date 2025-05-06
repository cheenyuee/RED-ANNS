#pragma once

#include "common.h"
#include "store/vertex.hpp"
#include "tsl/robin_map.h"
// #include "global.hpp"
// #include "unit.hpp"
// #include "logger2.hpp"
// #include <atomic>

class RDMA_Cache
{
public:
    size_t vertex_num, vertex_size;
    std::unique_ptr<char[]> cache_data = nullptr;
    std::vector<tsl::robin_map<local_id_t, char *>> cache_table;
    // uint64_t lookup_cnt = 0, hit_cnt = 0, total_ns = 0;

    RDMA_Cache(size_t vertex_size)
    {
        this->vertex_num = 0;
        this->vertex_size = vertex_size;
        this->cache_data.reset();
        this->cache_table.resize(Global::num_servers);
    }
    ~RDMA_Cache() {}

    /* 仅用于测试 */
    // void set_cache()
    // {
    //     size_t file_size;
    //     // this->cache_data.reset((char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1", file_size));
    //     this->cache_data.reset((char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1_remote_neighbor_order", file_size));
    //     size_t vertex_num = file_size / vertex_size;
    //     // std::cout << "inpute_server_1 count: " << vertex_num << std::endl;
    //     for (size_t dst_lid = 0; dst_lid < vertex_num; dst_lid++)
    //     {
    //         cache_table[1][dst_lid] = cache_data.get() + (uint64_t)dst_lid * vertex_size;
    //     }
    // }

    /* 仅用于测试 */
    // inline char *lookup(const ikey_t &key)
    // {
    //     server_id_t dst_sid = key.first;
    //     local_id_t dst_lid = key.second;
    //     uint64_t dst_off = (uint64_t)dst_lid * vertex_size;
    //     return cache_data.get() + dst_off;
    // }

    void set_cache(int sid, Mem *mem, const std::vector<ikey_t> &keys)
    {
        this->vertex_num = keys.size();
        this->cache_data.reset(new char[vertex_num * vertex_size]);
        for (size_t i = 0; i < vertex_num; i++)
        {
            server_id_t dst_sid = keys[i].first;
            local_id_t dst_lid = keys[i].second;
            uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

            // 忽略已经位于本地的vertex
            if ((int)dst_sid == sid)
            {
                logstream(LOG_DEBUG) << "#" << sid << ": try to cache local vertex." << LOG_endl;
            }

            int tid = 0;
            char *buf = mem->buffer(tid);
            uint64_t buf_sz = mem->buffer_size();
            ASSERT(vertex_size < buf_sz);

            RDMA &rdma = RDMA::get_rdma();
            int succ = rdma.dev->RdmaRead(tid, dst_sid, buf, vertex_size, dst_off);

            char *ptr = cache_data.get() + (uint64_t)i * vertex_size;
            std::memcpy(ptr, buf, vertex_size);
            cache_table[dst_sid][dst_lid] = ptr;
        }
    }

    inline char *lookup(const ikey_t &key)
    {
        // reinterpret_cast<std::atomic<uint64_t> &>(lookup_cnt).fetch_add(1);
        // auto start = std::chrono::high_resolution_clock::now();
        auto iter = cache_table[key.first].find(key.second);
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        // reinterpret_cast<std::atomic<uint64_t> &>(total_ns).fetch_add(duration);
        if (iter != cache_table[key.first].end())
        {
            // reinterpret_cast<std::atomic<uint64_t> &>(hit_cnt).fetch_add(1);
            return iter->second;
        }
        else
        {
            return nullptr;
        }
    }
};
