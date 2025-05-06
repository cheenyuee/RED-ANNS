#pragma once

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
// RDMA header file
#include "store/gstore.hpp"
// ANNS header file
#include "parameters.h"
#include "neighbor.h"
#include "distance.h"
#include "util.h"
#include "common.h"
#include "hungary.h"

#include "pq.h"
#include "index_bf.h"

class PQ
{
public:
    uint32_t num_pq_chunks;
    efanna2e::FixedChunkPQTable pq_table;
    uint8_t *pq_data = nullptr;
    std::vector<float *> thd_dist_vec;
    void load()
    {
        unsigned N = 100 * 1000 * 1000, Dim = 96;
        num_pq_chunks = Dim/4;
        std::string pq_pivots_path = "./data/deep100M_pq_pivots.fbin";
        std::string pq_comp_path = "./data/deep100M_pq_comp.fbin";

        efanna2e::alloc_aligned(((void **)&pq_data), N * num_pq_chunks * sizeof(uint8_t), 1);
        efanna2e::copy_aligned_data_from_file<uint8_t>(pq_comp_path.c_str(), pq_data, N, num_pq_chunks, num_pq_chunks);
        pq_table.load_pq_centroid_bin(pq_pivots_path.c_str(), num_pq_chunks);

        unsigned T = 16;
        thd_dist_vec.resize(T);
        for (unsigned i = 0; i < T; i++)
        {
            thd_dist_vec[i] = new float[num_pq_chunks * NUM_PQ_CENTROIDS];
        }
    }
    inline void inti_dist_vec(int tid, const float *query)
    {
        float *dist_vec = thd_dist_vec[tid];
        pq_table.populate_chunk_distances(query, dist_vec);
    }

    inline float compute_dist(int tid, unsigned base_id)
    {
        float *dist_vec = thd_dist_vec[tid];
        float dist = efanna2e::pq_dist_lookup_single(&pq_data[base_id * num_pq_chunks], num_pq_chunks, dist_vec);
        return dist;
    }
};

namespace numaann
{
    class Index
    {
    public:
        PQ pq;
        void test_compute(int tid, const float *query, uint64_t NUM_ITERATIONS, uint64_t batch_size);
        explicit Index(const Parameters &para);
        ~Index();

        /* load_coarse_clusters */
        size_t _coarse_clusters_num{0};
        std::unique_ptr<float[]> _coarse_clusters_data{nullptr};
        std::vector<unsigned> _coarse_clusters_label;
        void load_coarse_clusters();

        unsigned compute_closest_point(const float *base_data, size_t base_num, size_t base_dim, const float *query_point, const Distance *distance);
        unsigned compute_closest_coarse_cluster(const float *point);

        const Parameters _para;
        Distance *_distance = nullptr;
        size_t _dimension{0};

        /* load_learn_data */
        size_t _learn_num;
        std::unique_ptr<float[]> _learn_data = nullptr;
        void load_learn_data();

        /* load_learn_graph */
        std::vector<std::vector<unsigned>> _learn_graph;
        unsigned _learn_graph_R{0};
        unsigned _learn_graph_EP;
        void load_learn_graph();

        /* generate_learn_projection */
        std::vector<std::vector<unsigned>> _learn_projection_in_base;
        std::unique_ptr<unsigned[]> _learn_in_bucket = nullptr;
        std::unique_ptr<float[]> _learn_local_ratio = nullptr;
        void generate_learn_projection();

        unsigned beam_search_learn_graph(const float *query, size_t L);

        std::pair<unsigned, float> search_affinity(const float *query);
        std::pair<unsigned, float> search_affinity(unsigned learn_index_res);
        std::pair<unsigned, float> search_affinity(const std::vector<unsigned> &gt);

        void add_adaptive_ep(unsigned learn_index_res, unsigned adaptive_ep_num, std::vector<item_t> &init_ids);

        // 单机测试
        /* load_base_data */
        size_t _base_num;
        std::unique_ptr<float[]> _base_data = nullptr;
        void load_base_data();

        /* load_base_graph */
        std::vector<std::vector<unsigned>> _base_graph;
        unsigned _base_graph_R{0};
        unsigned _base_graph_EP;
        void load_base_graph();

        /* generate_base_index */
        size_t _element_num{0}, _element_size{0};
        size_t _data_offset{0}, _label_offset{0}, _neighbor_offset{0};
        unsigned _base_index_R{0};

        // 单机测试
        std::unique_ptr<char[]> _base_index = nullptr;
        unsigned _base_index_EP;
        void generate_base_index();
        void search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices, float *distances, common::QueryStats *stats = nullptr);

        // 伪分布式测试(实现数据划分)
        size_t bucket_count;
        std::vector<size_t> data_num;
        std::unique_ptr<unsigned[]> _base_in_bucket = nullptr; // _base_in_bucket[label]，这里的 label 指 _label_offset 的 label
        std::vector<unsigned> _base_to_lid;
        char **_memeory_buckets = nullptr;
        item_t _membkt_EP;
        void generate_base_index_on_buckets();
        void save_base_index_on_buckets();
        void search_base_index_on_buckets(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices);

        // 分布式测试
        GStore *_gstore = nullptr;
        void load_base_index_distributed(int sid, Mem *mem);
        void set_cache(float *query_data, unsigned query_num, const std::vector<unsigned> &query_bucket, uint32_t nthreads, size_t num_nodes_to_cache);
        void search_base_index_distributed(int tid, const float *query, std::vector<std::vector<uint32_t>> &access_count);
        void search_base_index_distributed(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices, unsigned learn_index_res, unsigned adaptive_ep_num, int relax, common::QueryStats *stats = nullptr);

        /* 单机下的_query_scratch */
        std::vector<diskann::InMemQueryScratch *> _query_scratch;
        void initialize_query_scratch(uint32_t num_threads, uint32_t search_l)
        {
            if (this->_base_index_R == 0)
                throw std::runtime_error("error@initialize_query_scratch: _base_index_R is 0.");
            for (uint32_t i = 0; i < num_threads; i++)
            {
                auto scratch = new diskann::InMemQueryScratch(search_l, this->_base_index_R);
                _query_scratch.push_back(scratch);
            }
        }

        /* 分布式下的_query_scratch */
        std::vector<dsmann::InMemQueryScratch *> _query_scratch_distributed;
        void initialize_query_scratch_distributed(uint32_t num_servers, uint32_t num_threads, uint32_t search_l)
        {
            if (this->_base_index_R == 0)
                throw std::runtime_error("error@initialize_query_scratch: _base_index_R is 0.");
            for (uint32_t i = 0; i < num_threads; i++)
            {
                auto scratch = new dsmann::InMemQueryScratch(search_l, this->_base_index_R, num_servers);
                _query_scratch_distributed.push_back(scratch);
            }
        }
    };

}
