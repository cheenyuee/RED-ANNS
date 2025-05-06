#include <boost/container/set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "efanna2e/index.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "visited_list_pool.h"
#include "pq.h"


namespace efanna2e {
using LockGuard = std::lock_guard<std::mutex>;
using SharedLockGuard = std::lock_guard<std::shared_mutex>;

class IndexBF : public Index {
   public:

    explicit IndexBF(const size_t dimension, const size_t n, Metric m, Index *initializer): Index(dimension, n, m), initializer_{initializer} {
        l2_distance_ = new DistanceL2();
        if (m == efanna2e::COSINE) {
            need_normalize = true;
        }
    }
    ~IndexBF(){}


    virtual void Save(const char *filename){}
    virtual void Load(const char *filename){}
    virtual void Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices, float *res_dists){}
    virtual void Build(size_t n, const float *data, const Parameters &parameters){}

    void InitVisitedListPool(uint32_t num_threads) { visited_list_pool_ = new VisitedListPool(num_threads, nd_); };

    // Load relevent data into index: _num_pq_chunks, _pq_data, _pq_table;
    void load_pq_structure(uint32_t num_points, uint32_t num_pq_chunks, std::string pq_pivots_file, std::string pq_compressed_vectors_file){
        _num_pq_chunks = num_pq_chunks;

        alloc_aligned(((void **)&_pq_data), num_points * num_pq_chunks * sizeof(uint8_t), 1);
        copy_aligned_data_from_file<uint8_t>(pq_compressed_vectors_file.c_str(), _pq_data,
                                            num_points, num_pq_chunks,
                                            num_pq_chunks);
        _pq_table.load_pq_centroid_bin(pq_pivots_file.c_str(), num_pq_chunks);
    }

    // Load Base data
    void LoadBaseData(std::string base_data_file, uint32_t &base_num, uint32_t &base_dim){
        efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
        _base_num = base_num, _base_dim = base_dim;

        float *base_data = nullptr;
        efanna2e::load_data<float>(base_data_file.c_str(), _base_num, _base_dim, base_data);
        float *aligned_base_data = efanna2e::data_align(base_data, base_num, base_dim);

        if (need_normalize) {
            std::cout << "Normalizing base data" << std::endl;
            for (size_t i = 0; i < base_num; ++i) {
                efanna2e::normalize<float>(aligned_base_data + i * (uint64_t)base_dim, (uint64_t)base_dim);
            }
        }
        _base_data = aligned_base_data;
    }


    // Brutal force with PQ
    void BFSearch(float *query, size_t k, size_t &qid, unsigned *indices, std::vector<float>& res_dists) {  
        // make preparation for pq based on pq_table
        // _pq_table.preprocess_query(query);  // Uncomment if centering query is required
        float *pq_dists = new float[_num_pq_chunks * NUM_PQ_CENTROIDS];
        _pq_table.populate_chunk_inner_products(query, pq_dists);

        float *dists = new float[_base_num];
        for (size_t i = 0; i < _base_num; ++i){
            dists[i] = pq_dist_lookup_single(_pq_data + i * _num_pq_chunks, _num_pq_chunks, pq_dists);
        }

        // Use a priority queue to find the top-k nearest neighbors
        std::priority_queue<std::pair<float, unsigned>> top_k;
        for (size_t i = 0; i < _base_num; i++) {
            if (top_k.size() < k) {
                top_k.emplace(dists[i], i);
            } 
            else if (dists[i] < top_k.top().first) {
                top_k.pop();
                top_k.emplace(dists[i], i);
            }
        }

        // Extract results from the priority queue
        res_dists.resize(k);
        for (int i = k - 1; i >= 0; --i) {
            res_dists[i] = top_k.top().first;
            indices[i] = top_k.top().second;
            top_k.pop();
        }

        // Clean up allocated memory
        delete[] pq_dists;
        delete[] dists;
    }



    public:
        Index *initializer_;
        VisitedListPool *visited_list_pool_{nullptr};
        bool need_normalize = false;

    protected:    
        Distance *l2_distance_;
        std::vector<std::mutex> locks_;
        
        // Basic parameters
        float* _base_data;
        uint32_t _base_num;
        uint32_t _base_dim;

        // PQ parameters
        size_t            _num_pq_chunks = 0;
        uint8_t           *_pq_data = nullptr;
        FixedChunkPQTable _pq_table;
};
}  // namespace efanna2e