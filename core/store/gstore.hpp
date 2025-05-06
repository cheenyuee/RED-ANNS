#pragma once

#include "mem.hpp"
#include "bind.hpp"
#include "config.hpp"
#include "store/vertex.hpp"
#include "store/cache.hpp"
#include <queue>
#include <boost/circular_buffer.hpp>
#include "tsl/robin_map.h"

// struct hop
// {
//     std::vector<item_t> item_batch;
//     std::vector<char *> vertex_batch;
//     std::vector<int> polls;
// };

class GStore
{
public:
    int sid;

    // int max_relax = 3;
    // std::vector<std::vector<hop>> thd_hops;

    std::vector<uint64_t> thd_cnt_io;
    std::vector<uint64_t> thd_sz_io;
    inline void print_io()
    {
        size_t cnt_io = 0, sz_io = 0;
        for (size_t tid = 0; tid < Global::num_threads; tid++)
        {
            cnt_io += thd_cnt_io[tid];
            sz_io += thd_sz_io[tid];
        }
        std::cout << "IO Cnt: " << cnt_io << std::endl;
        std::cout << "IO Size: " << sz_io << std::endl;
        if (cnt_io > 0)
            std::cout << "Avg Message Size: " << sz_io / cnt_io << std::endl;
        return;
    }

public:
    Mem *mem;
    size_t vertex_size;

    RDMA_Cache *rdma_cache = nullptr;

    std::vector<uint64_t> thd_buf_used_sz;
    // std::vector<std::vector<std::unordered_map<local_id_t, char *>>> thd_cache;
    std::vector<std::vector<tsl::robin_map<local_id_t, char *>>> thd_cache;
    std::vector<std::queue<std::vector<int>>> thd_polls;

    std::vector<std::vector<int>> thd_nids;
    std::vector<std::vector<char *>> thd_local_batch;
    std::vector<std::vector<uint64_t>> thd_off_batch;

public:
    GStore(int sid, Mem *mem, size_t vertex_size, size_t _R) : sid(sid), mem(mem), vertex_size(vertex_size)
    {
        // print gstore usage
        if (mem == nullptr)
            logstream(LOG_DEBUG) << "#" << sid << ": mem is nullptr" << LOG_endl;
        else
            logstream(LOG_INFO) << "#" << sid << ": Gstore = " << mem->kvstore_size() << " bytes" << LOG_endl;

        this->rdma_cache = new RDMA_Cache(vertex_size);
        /* 测试 rdma_cache */
        // size_t file_size;
        // char *tmp = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1", file_size);
        // delete[] tmp;
        // size_t vertex_num = file_size / vertex_size;
        // std::vector<ikey_t> keys;
        // for (size_t dst_lid = 0; dst_lid < vertex_num; dst_lid++)
        // {
        //     keys.push_back(std::make_pair<server_id_t, local_id_t>(1, dst_lid));
        // }
        // this->rdma_cache->set_cache(sid, mem, keys);

        // init thd_buf
        thd_buf_used_sz.resize(Global::num_threads, 0);

        // init thd_cache
        this->thd_cache.resize(Global::num_threads);
        for (int tid = 0; tid < Global::num_threads; tid++)
            this->thd_cache[tid].resize(Global::num_servers);

        this->thd_polls.resize(Global::num_threads);

        this->thd_cnt_io.resize(Global::num_threads, 0);
        this->thd_sz_io.resize(Global::num_threads, 0);

        this->thd_nids.resize(Global::num_threads);
        this->thd_local_batch.resize(Global::num_threads);
        this->thd_off_batch.resize(Global::num_threads);
        for (int tid = 0; tid < Global::num_threads; tid++)
        {
            thd_nids[tid].reserve(_R);
            thd_local_batch[tid].reserve(_R);
            thd_off_batch[tid].reserve(_R);
        }

        // this->thd_hops.resize(Global::num_threads);
        // for (int tid = 0; tid < Global::num_threads; tid++)
        // {
        //     thd_hops[tid].resize(max_relax + 1);
        //     for (size_t i = 0; i < max_relax + 1; i++)
        //     {
        //         thd_hops[tid][i].item_batch.reserve(_R);
        //         thd_hops[tid][i].vertex_batch.reserve(_R);
        //         thd_hops[tid][i].polls.reserve(Global::num_servers);
        //     }
        // }
    }

    inline void set_cache(const std::vector<std::vector<uint32_t>> &access_count, std::vector<size_t> data_num, size_t num_nodes_to_cache)
    {
        std::vector<std::pair<uint32_t, item_t>> frequency;
        uint64_t total_remote_access = 0;
        for (int dsid = 0; dsid < Global::num_servers; dsid++)
        {
            if (dsid == sid)
                continue;
            for (size_t dlid = 0; dlid < data_num[dsid]; dlid++)
            {
                total_remote_access += access_count[dsid][dlid];
                frequency.push_back(std::make_pair(access_count[dsid][dlid], std::make_pair(dsid, dlid)));
            }
        }
        std::sort(frequency.begin(), frequency.end(), std::greater<std::pair<uint32_t, item_t>>());

        // for (size_t i = num_nodes_to_cache - 100; i < num_nodes_to_cache; i++)
        // {
        //     std::cout << "freq: " << frequency[i].first << " (" << (int)frequency[i].second.first << "," << frequency[i].second.second << ")" << std::endl;
        // }

        std::vector<ikey_t> keys;
        uint64_t cached_remote_access = 0;
        for (size_t i = 0; i < frequency.size(); i++)
        {
            if (keys.size() >= num_nodes_to_cache)
                break;
            if(frequency[i].first == 0)
                break;
            cached_remote_access += frequency[i].first;
            keys.push_back(frequency[i].second);
        }
        logstream(LOG_EMPH) << "#" << sid << ": try to cache num: " << num_nodes_to_cache << ", actual: " << keys.size()
                            << ", cached_remote_access: " << (float)cached_remote_access / total_remote_access * 100 << "%" << LOG_endl;
        this->rdma_cache->set_cache(sid, mem, keys);
    }

    /*
     * 每次线程执行 query 前调用
     */
    inline void reset_thd_ctx(int tid)
    {
        // reset thd_buf
        this->thd_buf_used_sz[tid] = 0;
        // reset thd_cache
        for (size_t dst_sid = 0; dst_sid < Global::num_servers; dst_sid++)
            this->thd_cache[tid][dst_sid].clear();
    }

    inline void update_thd_cache(int tid, item_t item, char *ptr)
    {
        if (item.first != sid) // when to cache
            this->thd_cache[tid][item.first][item.second] = ptr;
    }

    /* Get vertex of given key. */
    inline char *get_vertex(int tid, ikey_t key)
    {
        assert(Global::use_rdma);
        assert(tid < Global::num_threads);

        server_id_t dst_sid = key.first;
        local_id_t dst_lid = key.second;
        uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

        // Get local vertex dirctly（如果用RdmaRead读本地内存会导致性能损失）
        if ((int)dst_sid == sid)
            return mem->kvstore() + dst_off;

        // check thd_cache
        auto iter = thd_cache[tid][dst_sid].find(dst_lid);
        if (iter != thd_cache[tid][dst_sid].end())
            return iter->second;

        logstream(LOG_ERROR) << "thd_cache miss." << LOG_endl;
        throw std::runtime_error("thd_cache miss.");

        // Get remote vertex by RDMA
        char *buf = mem->buffer(tid);
        uint64_t buf_sz = mem->buffer_size();
        buf += thd_buf_used_sz[tid];

        char *buf_vertex = buf;
        buf += vertex_size;
        thd_buf_used_sz[tid] += vertex_size;
        ASSERT(thd_buf_used_sz[tid] < buf_sz); // enough space to host the vertices

        /* 这里可能有bug，当有多个wr在SQ中时，虽然RdmaRead采用同步I/O，但无法保证poll到的wc一定对应这里提交的wr */
        RDMA &rdma = RDMA::get_rdma();
        int succ = rdma.dev->RdmaRead(tid, dst_sid, buf_vertex, vertex_size, dst_off);
        return buf_vertex;
    }

    /*
     * 没有优化的 get_vertex_batch
     */
    // inline void get_vertex_batch(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res)
    // {
    //     assert(Global::use_rdma);
    //     assert(tid < Global::num_threads);

    //     char *buf = mem->buffer(tid);
    //     uint64_t buf_sz = mem->buffer_size();

    //     uint64_t buf_blk_sz = thd_buf_blk_sz[tid];
    //     uint64_t buf_blk_idx = thd_buf_blk_idx[tid];
    //     thd_buf_blk_idx[tid]++;

    //     uint64_t buf_blk_num = buf_sz / buf_blk_sz;
    //     ASSERT(buf_blk_idx < buf_blk_num);             // enough space to host the vertices
    //     ASSERT(batch_size * vertex_size < buf_blk_sz); // enough space to host the vertices
    //     buf += buf_blk_idx * buf_blk_sz;

    //     RDMA &rdma = RDMA::get_rdma();
    //     for (size_t i = 0; i < batch_size; i++)
    //     {
    //         server_id_t dst_sid = key_batch[i].first;
    //         local_id_t dst_lid = key_batch[i].second;
    //         uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //         // Get local vertex dirctly
    //         if ((int)dst_sid == sid)
    //         {
    //             res.push_back(mem->kvstore() + dst_off);
    //             continue;
    //         }

    //         // Get remote vertex by RDMA
    //         char *buf_vertex = buf;
    //         buf += vertex_size;
    //         int succ = rdma.dev->RdmaRead(tid, dst_sid, buf_vertex, vertex_size, dst_off);
    //         res.push_back(buf_vertex);
    //     }
    //     return;
    // }

    /*
     * RdmaReadBatch 并发I/O优化后的 get_vertex_batch
     */
    inline void get_vertex_batch(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res, bool async = false)
    {
        assert(Global::use_rdma);
        assert(tid < Global::num_threads);

        char *buf = mem->buffer(tid);
        uint64_t buf_sz = mem->buffer_size();
        buf += thd_buf_used_sz[tid];

        RDMA &rdma = RDMA::get_rdma();
        std::vector<int> &nids = thd_nids[tid];
        std::vector<char *> &local_batch = thd_local_batch[tid];
        std::vector<uint64_t> &off_batch = thd_off_batch[tid];
        nids.clear();
        local_batch.clear();
        off_batch.clear();

        for (size_t i = 0; i < batch_size; i++)
        {
            server_id_t dst_sid = key_batch[i].first;
            local_id_t dst_lid = key_batch[i].second;
            uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

            // Get local vertex dirctly
            if ((int)dst_sid == sid)
            {
                res.push_back(mem->kvstore() + dst_off);
                continue;
            }

            // check rdma_cache
            char *cached_vertex = rdma_cache->lookup(key_batch[i]);
            if (cached_vertex)
            {
                res.push_back(cached_vertex);
                continue;
            }
            // throw std::runtime_error("error:get_vertex_batch.");

            // Get remote vertex by RDMA
            char *buf_vertex = buf;
            buf += vertex_size;
            thd_buf_used_sz[tid] += vertex_size;
            // int succ = rdma.dev->RdmaRead(tid, dst_sid, buf_vertex, vertex_size, dst_off);
            res.push_back(buf_vertex);

            nids.push_back(dst_sid);
            local_batch.push_back(buf_vertex);
            off_batch.push_back(dst_off);
        }
        assert(thd_buf_used_sz[tid] < buf_sz); // enough space to host the vertices

        // need I/O
        if (async)
        {
            // 异步I/O
            std::vector<int> polls;
            polls.resize(Global::num_servers, 0);
            if (off_batch.size() != 0)
            {
                // int succ = rdma.dev->RdmaReadBatch_Async_Send(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size(), polls); // 1.1
                // int succ = rdma.dev->RdmaReadBatchDoorbell_Async_Send(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size(), polls); // 1.2
                // int succ = rdma.dev->RdmaReadBatchUnsignal_Async_Send(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size(), polls); // 2.1
                int succ = rdma.dev->RdmaReadBatchDoorbellUnsignal_Async_Send(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size(), polls); // 2.2（性能最好）
            }
            thd_polls[tid].push(polls);
        }
        else
        {
            // 同步I/O
            if (off_batch.size() != 0)
            {
                int succ = rdma.dev->RdmaReadBatch(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size());
                // int succ = rdma.dev->RdmaReadBatchDoorbell(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size());
                // int succ = rdma.dev->RdmaReadBatchUnsignal(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size());
            }
        }
        return;
    }

    /* 新修改的get_vertex_batch_merged */
    // inline void get_vertex_batch_merged(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res, bool async = false)
    // {
    //     assert(Global::use_rdma);
    //     assert(tid < Global::num_threads);

    //     char *buf = mem->buffer(tid);
    //     uint64_t buf_sz = mem->buffer_size();
    //     buf += thd_buf_used_sz[tid];

    //     RDMA &rdma = RDMA::get_rdma();
    //     std::vector<int> &nids = thd_nids[tid];
    //     std::vector<char *> &local_batch = thd_local_batch[tid];
    //     std::vector<uint64_t> &off_batch = thd_off_batch[tid];
    //     nids.clear();
    //     local_batch.clear();
    //     off_batch.clear();

    //     std::vector<uint64_t> length_batch;
    //     length_batch.reserve(55);

    //     // 要求请求地址升序
    //     // for (size_t i = 1; i < batch_size; i++)
    //     // {
    //     //     ASSERT(key_batch[i - 1].first <= key_batch[i].first);
    //     //     if (key_batch[i - 1].first == key_batch[i].first)
    //     //         ASSERT(key_batch[i - 1].second < key_batch[i].second);
    //     // }
    //     /*实现一*/
    //     ikey_t key_last_read;
    //     char *buf_last_read = nullptr;
    //     int cnt_last_read;
    //     for (size_t i = 0; i < batch_size; i++)
    //     {
    //         server_id_t dst_sid = key_batch[i].first;
    //         local_id_t dst_lid = key_batch[i].second;
    //         uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //         // Get local vertex dirctly
    //         if ((int)dst_sid == sid)
    //         {
    //             res.push_back(mem->kvstore() + dst_off);
    //             continue;
    //         }

    //         // Get remote vertex by RDMA
    //         if (buf_last_read != nullptr and
    //             dst_sid == key_last_read.first and
    //             dst_lid > key_last_read.second and
    //             dst_lid - key_last_read.second < cnt_last_read)
    //         {
    //             res.push_back(buf_last_read + (dst_lid - key_last_read.second) * vertex_size);
    //             continue;
    //         }

    //         // 检查有没有请求能合并(无读放大)(有4%性能提升)
    //         cnt_last_read = 1;
    //         for (size_t j = i + 1; j < batch_size; j++)
    //         {
    //             if (key_batch[j].first == dst_sid and
    //                 key_batch[j].second > dst_lid and
    //                 key_batch[j].second - dst_lid == j - i)
    //                 cnt_last_read = 1 + key_batch[j].second - dst_lid;
    //             else
    //                 break;
    //         }

    //         key_last_read = key_batch[i];
    //         buf_last_read = buf;
    //         buf += cnt_last_read * vertex_size;
    //         thd_buf_used_sz[tid] += cnt_last_read * vertex_size;

    //         res.push_back(buf_last_read);

    //         nids.push_back(dst_sid);
    //         local_batch.push_back(buf_last_read);
    //         length_batch.push_back(cnt_last_read * vertex_size);
    //         off_batch.push_back(dst_off);

    //         // 共享原子变量会严重影响多线程性能（仅用于统计）
    //         thd_cnt_io[tid]++;
    //         thd_sz_io[tid] += cnt_last_read * vertex_size;
    //     }
    //     assert(thd_buf_used_sz[tid] < buf_sz); // enough space to host the vertices

    //     // need I/O
    //     if (async)
    //     {
    //         // 异步I/O
    //         std::vector<int> polls;
    //         polls.resize(Global::num_servers, 0);
    //         if (off_batch.size() != 0)
    //         {
    //             int succ = rdma.dev->RdmaReadBatchDoorbellUnsignal_Async_Send(tid, nids.data(), local_batch.data(), length_batch.data(), off_batch.data(), off_batch.size(), polls);
    //         }
    //         thd_polls[tid].push(polls);
    //     }
    //     else
    //     {
    //         throw std::runtime_error("还未实现");
    //     }
    //     return;
    // }

    inline void get_vertex_batch_wait(int tid)
    {
        if (thd_polls[tid].empty())
            logstream(LOG_ERROR) << "thd_polls is empty." << LOG_endl;
        std::vector<int> &polls = thd_polls[tid].front();
        RDMA &rdma = RDMA::get_rdma();
        // int succ = rdma.dev->RdmaReadBatch_Async_Wait(tid, polls); // 1
        int succ = rdma.dev->RdmaReadBatchUnsignal_Async_Wait(tid, polls); // 2
        thd_polls[tid].pop();
    }

    // inline void get_vertex_batch_hop(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res, std::vector<int> &polls)
    // {
    //     assert(Global::use_rdma);
    //     assert(tid < Global::num_threads);

    //     char *buf = mem->buffer(tid);
    //     uint64_t buf_sz = mem->buffer_size();
    //     buf += thd_buf_used_sz[tid];

    //     RDMA &rdma = RDMA::get_rdma();
    //     std::vector<int> &nids = thd_nids[tid];
    //     std::vector<char *> &local_batch = thd_local_batch[tid];
    //     std::vector<uint64_t> &off_batch = thd_off_batch[tid];
    //     nids.clear();
    //     local_batch.clear();
    //     off_batch.clear();

    //     for (size_t i = 0; i < batch_size; i++)
    //     {
    //         server_id_t dst_sid = key_batch[i].first;
    //         local_id_t dst_lid = key_batch[i].second;
    //         uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //         // Get local vertex dirctly
    //         if ((int)dst_sid == sid)
    //         {
    //             res.push_back(mem->kvstore() + dst_off);
    //             continue;
    //         }

    //         // check rdma_cache
    //         char *cached_vertex = rdma_cache->lookup(key_batch[i]);
    //         if (cached_vertex)
    //         {
    //             res.push_back(cached_vertex);
    //             continue;
    //         }
    //         // throw std::runtime_error("error:get_vertex_batch.");

    //         // Get remote vertex by RDMA
    //         char *buf_vertex = buf;
    //         buf += vertex_size;
    //         thd_buf_used_sz[tid] += vertex_size;
    //         // int succ = rdma.dev->RdmaRead(tid, dst_sid, buf_vertex, vertex_size, dst_off);
    //         res.push_back(buf_vertex);

    //         nids.push_back(dst_sid);
    //         local_batch.push_back(buf_vertex);
    //         off_batch.push_back(dst_off);
    //     }
    //     assert(thd_buf_used_sz[tid] < buf_sz); // enough space to host the vertices

    //     // 异步I/O
    //     ASSERT(polls.empty());
    //     polls.resize(Global::num_servers, 0);
    //     if (off_batch.size() != 0)
    //         int succ = rdma.dev->RdmaReadBatchDoorbellUnsignal_Async_Send(tid, nids.data(), local_batch.data(), vertex_size, off_batch.data(), off_batch.size(), polls); // 2.2（性能最好）
    //     return;
    // }

    // inline void get_vertex_batch_wait_hop(int tid, std::vector<int> &polls)
    // {
    //     ASSERT(!polls.empty());
    //     RDMA &rdma = RDMA::get_rdma();
    //     int succ = rdma.dev->RdmaReadBatchUnsignal_Async_Wait(tid, polls); // 2
    //     polls.clear();
    // }

    /*
     * reorder 优化后的 get_vertex_batch：串行Read，但是减少Read次数
     * 减少 20% 的io次数，性能提升了 13%
     * max_vertex_per_read = 1，cnt_io：17972188，qps：185
     * max_vertex_per_read = 32，cnt_io：14680707，qps：210
     */
    // size_t cnt_io{0};
    // inline void get_vertex_batch(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res)
    // {
    //     size_t max_vertex_per_read = 32;

    //     assert(Global::use_rdma);
    //     assert(tid < Global::num_threads);

    //     char *buf = mem->buffer(tid);
    //     uint64_t buf_sz = mem->buffer_size();
    //     assert(buf_sz > (uint64_t)batch_size * max_vertex_per_read * vertex_size); // enough space to host the vertices
    //     RDMA &rdma = RDMA::get_rdma();

    //     std::vector<int> keys_ids;
    //     for (int req_dst_sid = 0; req_dst_sid < Global::num_servers; req_dst_sid++)
    //     {
    //         keys_ids.clear();
    //         for (size_t i = 0; i < batch_size; i++)
    //         {
    //             server_id_t dst_sid = key_batch[i].first;
    //             if ((int)dst_sid != req_dst_sid)
    //                 continue;
    //             keys_ids.push_back(i);
    //         }
    //         for (size_t i = 0; i < keys_ids.size(); i++)
    //         {
    //             if (i < keys_ids.size() - 1)
    //                 ASSERT(key_batch[keys_ids[i]].second < key_batch[keys_ids[i + 1]].second);
    //         }
    //         ikey_t key_last_read;
    //         int cnt_last_read;
    //         char *buf_last_read = nullptr;
    //         for (size_t i = 0; i < keys_ids.size(); i++)
    //         {
    //             ikey_t key = key_batch[keys_ids[i]];
    //             server_id_t dst_sid = key.first;
    //             local_id_t dst_lid = key.second;
    //             uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //             // Get local vertex dirctly
    //             if ((int)dst_sid == sid)
    //             {
    //                 res.push_back(mem->kvstore() + dst_off);
    //                 continue;
    //             }

    //             // Get remote vertex by RDMA
    //             if (buf_last_read != nullptr and dst_lid - key_last_read.second > 0 and dst_lid - key_last_read.second < cnt_last_read)
    //             {
    //                 res.push_back(buf_last_read + (dst_lid - key_last_read.second) * vertex_size);
    //                 continue;
    //             }

    //             cnt_last_read = 1;
    //             for (size_t j = i + 1; j < keys_ids.size(); j++)
    //             {
    //                 if (key_batch[keys_ids[j]].second - dst_lid > 0 and key_batch[keys_ids[j]].second - dst_lid < max_vertex_per_read)
    //                     cnt_last_read = 1 + key_batch[keys_ids[j]].second - dst_lid;
    //                 else
    //                     break;
    //             }

    //             key_last_read = key;
    //             buf_last_read = buf;
    //             buf += cnt_last_read * vertex_size;
    //             int succ = rdma.dev->RdmaRead(tid, dst_sid, buf_last_read, cnt_last_read * vertex_size, dst_off);
    //             res.push_back(buf_last_read);
    //             cnt_io++;
    //         }
    //     }
    //     return;
    // }

    // 尝试通过 reorder 来减少请求数量
    // 需要保持 key_batch 和 res 中的数据项是对应的
    // size_t cnt_io{0};
    // size_t sz_io{0};
    // inline void get_vertex_batch_merged(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res, bool async = false)
    // {
    //     assert(Global::use_rdma);
    //     assert(tid < Global::num_threads);

    //     char *buf = mem->buffer(tid);
    //     uint64_t buf_sz = mem->buffer_size();

    //     uint64_t buf_blk_sz = thd_buf_blk_sz[tid];
    //     uint64_t buf_blk_idx = thd_buf_blk_idx[tid];
    //     thd_buf_blk_idx[tid]++;

    //     uint64_t buf_blk_num = buf_sz / buf_blk_sz;
    //     if (buf_blk_idx >= buf_blk_num)
    //     {
    //         logstream(LOG_ERROR) << "#" << sid << ": buf_blk_idx = " << buf_blk_idx << LOG_endl;
    //         logstream(LOG_ERROR) << "#" << sid << ": buf_blk_num = " << buf_blk_num << LOG_endl;
    //     }
    //     ASSERT(buf_blk_idx < buf_blk_num);             // enough space to host the vertices
    //     ASSERT(batch_size * vertex_size < buf_blk_sz); // enough space to host the vertices
    //     buf += buf_blk_idx * buf_blk_sz;

    //     RDMA &rdma = RDMA::get_rdma();
    //     std::vector<int> nids;
    //     std::vector<char *> local_batch;
    //     std::vector<uint64_t> off_batch;
    //     std::vector<uint64_t> length_batch;
    //     // 提前分配内存，避免vector动态分配
    //     nids.reserve(122);
    //     local_batch.reserve(122);
    //     off_batch.reserve(122);
    //     length_batch.reserve(122);

    //     // std::cout << "@@@@" << std::endl;
    //     size_t max_vertex_per_read = 32; // 调整这个参数
    //     if (batch_size * vertex_size * max_vertex_per_read > buf_blk_sz)
    //     {
    //         logstream(LOG_ERROR) << "#" << sid << ": batch_size = " << batch_size << LOG_endl;
    //         logstream(LOG_ERROR) << "#" << sid << ": vertex_size = " << vertex_size << LOG_endl;
    //         logstream(LOG_ERROR) << "#" << sid << ": max_vertex_per_read = " << max_vertex_per_read << LOG_endl;
    //         logstream(LOG_ERROR) << "#" << sid << ": buf_blk_sz = " << buf_blk_sz << LOG_endl;
    //     }
    //     ASSERT(batch_size * vertex_size * max_vertex_per_read < buf_blk_sz); // enough space to host the vertices

    //     // 要求请求地址升序
    //     // for (size_t i = 1; i < batch_size; i++)
    //     // {
    //     //     ASSERT(key_batch[i - 1].first <= key_batch[i].first);
    //     //     if (key_batch[i - 1].first == key_batch[i].first)
    //     //         ASSERT(key_batch[i - 1].second < key_batch[i].second);
    //     // }
    //     /*实现一*/
    //     ikey_t key_last_read;
    //     char *buf_last_read = nullptr;
    //     int cnt_last_read;
    //     for (size_t i = 0; i < batch_size; i++)
    //     {
    //         server_id_t dst_sid = key_batch[i].first;
    //         local_id_t dst_lid = key_batch[i].second;
    //         uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //         // Get local vertex dirctly
    //         if ((int)dst_sid == sid)
    //         {
    //             res.push_back(mem->kvstore() + dst_off);
    //             continue;
    //         }

    //         // Get remote vertex by RDMA
    //         if (buf_last_read != nullptr and
    //             dst_sid == key_last_read.first and
    //             dst_lid > key_last_read.second and
    //             dst_lid - key_last_read.second < cnt_last_read)
    //         {
    //             res.push_back(buf_last_read + (dst_lid - key_last_read.second) * vertex_size);
    //             continue;
    //         }

    //         // 检查有没有请求能合并(有读放大)
    //         // cnt_last_read = 1;
    //         // for (size_t j = i + 1; j < batch_size; j++)
    //         // {
    //         //     if (key_batch[j].first == dst_sid and
    //         //         key_batch[j].second > dst_lid and
    //         //         key_batch[j].second - dst_lid < max_vertex_per_read)
    //         //         cnt_last_read = 1 + key_batch[j].second - dst_lid;
    //         //     else
    //         //         break;
    //         // }
    //         // 检查有没有请求能合并(无读放大)(有4%性能提升，max_vertex_per_read再设大一点？)
    //         cnt_last_read = 1;
    //         for (size_t j = i + 1; j < batch_size; j++)
    //         {
    //             if (key_batch[j].first == dst_sid and
    //                 key_batch[j].second > dst_lid and
    //                 key_batch[j].second - dst_lid == j - i)
    //                 cnt_last_read = 1 + key_batch[j].second - dst_lid;
    //             else
    //                 break;
    //         }

    //         key_last_read = key_batch[i];
    //         buf_last_read = buf;
    //         buf += cnt_last_read * vertex_size;

    //         res.push_back(buf_last_read);

    //         nids.push_back(dst_sid);
    //         local_batch.push_back(buf_last_read);
    //         length_batch.push_back(cnt_last_read * vertex_size);
    //         off_batch.push_back(dst_off);
    //         cnt_io++;
    //         sz_io += cnt_last_read * vertex_size;
    //     }
    //     /*实现二*/
    //     // res.resize(batch_size);
    //     // std::vector<int> keys_ids;
    //     // for (int req_dst_sid = 0; req_dst_sid < Global::num_servers; req_dst_sid++)
    //     // {
    //     //     // 检查相同目的地的请求
    //     //     keys_ids.clear();
    //     //     for (size_t i = 0; i < batch_size; i++)
    //     //     {
    //     //         server_id_t dst_sid = key_batch[i].first;
    //     //         if ((int)dst_sid == req_dst_sid)
    //     //             keys_ids.push_back(i);
    //     //     }
    //     //     ikey_t key_last_read;
    //     //     char *buf_last_read = nullptr;
    //     //     int cnt_last_read;
    //     //     for (size_t i = 0; i < keys_ids.size(); i++)
    //     //     {
    //     //         ikey_t key = key_batch[keys_ids[i]];
    //     //         server_id_t dst_sid = key.first;
    //     //         local_id_t dst_lid = key.second;
    //     //         uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //     //         // Get local vertex dirctly
    //     //         if ((int)dst_sid == sid)
    //     //         {
    //     //             res[keys_ids[i]] = mem->kvstore() + dst_off;
    //     //             continue;
    //     //         }

    //     //         // Get remote vertex by RDMA
    //     //         if (buf_last_read != nullptr and dst_lid - key_last_read.second > 0 and dst_lid - key_last_read.second < cnt_last_read)
    //     //         {
    //     //             res[keys_ids[i]] = buf_last_read + (dst_lid - key_last_read.second) * vertex_size;
    //     //             continue;
    //     //         }

    //     //         // 检查有没有请求能合并
    //     //         cnt_last_read = 1;
    //     //         for (size_t j = i + 1; j < keys_ids.size(); j++)
    //     //         {
    //     //             if (key_batch[keys_ids[j]].second - dst_lid > 0 and key_batch[keys_ids[j]].second - dst_lid < max_vertex_per_read)
    //     //                 cnt_last_read = 1 + key_batch[keys_ids[j]].second - dst_lid;
    //     //             else
    //     //                 break;
    //     //         }

    //     //         key_last_read = key;
    //     //         buf_last_read = buf;
    //     //         buf += cnt_last_read * vertex_size;
    //     //         // int succ = rdma.dev->RdmaRead(tid, dst_sid, buf_last_read, cnt_last_read * vertex_size, dst_off);
    //     //         res[keys_ids[i]] = buf_last_read;

    //     //         nids.push_back(dst_sid);
    //     //         local_batch.push_back(buf_last_read);
    //     //         length_batch.push_back(cnt_last_read * vertex_size);
    //     //         off_batch.push_back(dst_off);
    //     //     }
    //     // }

    //     // need I/O
    //     if (async)
    //     {
    //         // 异步I/O
    //         std::vector<int> polls;
    //         polls.resize(Global::num_servers, 0);
    //         if (off_batch.size() != 0)
    //         {
    //             int succ = rdma.dev->RdmaReadBatchDoorbellUnsignal_Async_Send(tid, nids.data(), local_batch.data(), length_batch.data(), off_batch.data(), off_batch.size(), polls);
    //         }
    //         thd_polls[tid].push(polls);
    //     }
    //     else
    //     {
    //         throw std::runtime_error("还未实现");
    //     }
    //     return;
    // }

    // inline void get_vertex_batch_merged(int tid, ikey_t key_batch[], size_t batch_size, std::vector<char *> &res, bool async = false)
    // {
    //     assert(Global::use_rdma);
    //     assert(tid < Global::num_threads);

    //     char *buf = mem->buffer(tid);
    //     uint64_t buf_sz = mem->buffer_size();

    //     uint64_t buf_blk_sz = thd_buf_blk_sz[tid];
    //     uint64_t buf_blk_idx = thd_buf_blk_idx[tid];
    //     thd_buf_blk_idx[tid]++;

    //     uint64_t buf_blk_num = buf_sz / buf_blk_sz;
    //     if (buf_blk_idx >= buf_blk_num)
    //     {
    //         logstream(LOG_ERROR) << "#" << sid << ": buf_blk_idx = " << buf_blk_idx << LOG_endl;
    //         logstream(LOG_ERROR) << "#" << sid << ": buf_blk_num = " << buf_blk_num << LOG_endl;
    //     }
    //     ASSERT(buf_blk_idx < buf_blk_num);             // enough space to host the vertices
    //     ASSERT(batch_size * vertex_size < buf_blk_sz); // enough space to host the vertices
    //     buf += buf_blk_idx * buf_blk_sz;

    //     RDMA &rdma = RDMA::get_rdma();
    //     std::vector<int> nids;
    //     std::vector<char *> local_batch;
    //     std::vector<uint64_t> off_batch;
    //     std::vector<uint64_t> length_batch;
    //     // 提前分配内存，避免vector动态分配
    //     nids.reserve(122);
    //     local_batch.reserve(122);
    //     off_batch.reserve(122);
    //     length_batch.reserve(122);

    //     // 要求请求地址升序
    //     // for (size_t i = 1; i < batch_size; i++)
    //     // {
    //     //     ASSERT(key_batch[i - 1].first <= key_batch[i].first);
    //     //     if (key_batch[i - 1].first == key_batch[i].first)
    //     //         ASSERT(key_batch[i - 1].second < key_batch[i].second);
    //     // }
    //     /*实现一*/
    //     ikey_t key_last_read;
    //     char *buf_last_read = nullptr;
    //     int cnt_last_read;
    //     for (size_t i = 0; i < batch_size; i++)
    //     {
    //         server_id_t dst_sid = key_batch[i].first;
    //         local_id_t dst_lid = key_batch[i].second;
    //         uint64_t dst_off = (uint64_t)dst_lid * vertex_size;

    //         // Get local vertex dirctly
    //         if ((int)dst_sid == sid)
    //         {
    //             res.push_back(mem->kvstore() + dst_off);
    //             continue;
    //         }

    //         // Get remote vertex by RDMA
    //         if (buf_last_read != nullptr and
    //             dst_sid == key_last_read.first and
    //             dst_lid > key_last_read.second and
    //             dst_lid - key_last_read.second < cnt_last_read)
    //         {
    //             res.push_back(buf_last_read + (dst_lid - key_last_read.second) * vertex_size);
    //             continue;
    //         }

    //         // 检查有没有请求能合并(无读放大)(有4%性能提升)
    //         cnt_last_read = 1;
    //         for (size_t j = i + 1; j < batch_size; j++)
    //         {
    //             if (key_batch[j].first == dst_sid and
    //                 key_batch[j].second > dst_lid and
    //                 key_batch[j].second - dst_lid == j - i)
    //                 cnt_last_read = 1 + key_batch[j].second - dst_lid;
    //             else
    //                 break;
    //         }

    //         key_last_read = key_batch[i];
    //         buf_last_read = buf;
    //         buf += cnt_last_read * vertex_size;

    //         res.push_back(buf_last_read);

    //         nids.push_back(dst_sid);
    //         local_batch.push_back(buf_last_read);
    //         length_batch.push_back(cnt_last_read * vertex_size);
    //         off_batch.push_back(dst_off);

    //         // 共享原子变量会严重影响多线程性能（仅用于统计）
    //         thd_cnt_io[tid]++;
    //         thd_sz_io[tid] += cnt_last_read * vertex_size;
    //     }

    //     // need I/O
    //     if (async)
    //     {
    //         // 异步I/O
    //         std::vector<int> polls;
    //         polls.resize(Global::num_servers, 0);
    //         if (off_batch.size() != 0)
    //         {
    //             int succ = rdma.dev->RdmaReadBatchDoorbellUnsignal_Async_Send(tid, nids.data(), local_batch.data(), length_batch.data(), off_batch.data(), off_batch.size(), polls);
    //         }
    //         thd_polls[tid].push(polls);
    //     }
    //     else
    //     {
    //         throw std::runtime_error("还未实现");
    //     }
    //     return;
    // }
};