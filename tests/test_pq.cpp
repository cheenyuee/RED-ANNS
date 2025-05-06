#include <pq.h>
#include <iostream>
#include <common.h>
#include <index_bf.h>

int main(int argc, char *argv[])
{
    // std::string base_file = "/ann/data/deep100M/base.1B.fbin.crop_nb_100000000";
    std::string base_file = "/home/cy/data/deep10M/deep10M.fbin";
    // std::string pq_pivots_path = "/ann/data/deep100M/pq_pivots.fbin";
    std::string pq_pivots_path = "/home/csp/deep10M_pq_pivots.fbin";
    // std::string pq_comp_path = "/ann/data/deep100M/pq_comp.fbin";
    std::string pq_comp_path = "/home/csp/deep10M_pq_comp.fbin";
    // std::string query_file = "/ann/data/deep100M/query.public.10K.fbin";
    std::string query_file = "/home/cy/data/deep10M/deep10M_queries.fbin";
    std::string gt_file = "/home/cy/data/deep10M/deep10M_groundtruth.ibin";

    unsigned N = 10 * 1000 * 1000, Dim = 96;
    uint32_t num_pq_chunks = Dim;

    // float *base_data = common::read_data(base_file, "bin", N, Dim);
    // printf("N: %u, Dim: %u\n", N, Dim);
    efanna2e::Metric metric = efanna2e::Metric::L2;
    efanna2e::generate_quantized_data<float>(base_file, pq_pivots_path, pq_comp_path, metric, 0.2, num_pq_chunks);

    efanna2e::FixedChunkPQTable pq_table;
    uint8_t *pq_data = nullptr;

    efanna2e::alloc_aligned(((void **)&pq_data), N * num_pq_chunks * sizeof(uint8_t), 1);
    // 加载压缩向量
    efanna2e::copy_aligned_data_from_file<uint8_t>(pq_comp_path.c_str(), pq_data, N, num_pq_chunks, num_pq_chunks);
    // 加载centroids
    pq_table.load_pq_centroid_bin(pq_pivots_path.c_str(), num_pq_chunks);

    unsigned qN, qDim;
    float *query_data = common::read_data(query_file, "bin", qN, qDim);
    printf("qN: %u, qDim: %u\n", qN, qDim);
    qN = 100;

    std::vector<std::vector<unsigned>> res_indices;
    res_indices.reserve(qN);
    std::vector<std::pair<float, unsigned>> res;
    res.reserve(N);
    float *dist_vec = new float[num_pq_chunks * NUM_PQ_CENTROIDS];
    for (unsigned i = 0; i < qN; i++)
    {
        float *q = query_data + i * qDim;
        // 根据query计算距离
        pq_table.populate_chunk_distances(q, dist_vec);
        res.clear();
        printf("Processing %u\n", i);
        // 遍历每个压缩向量
        for (unsigned j = 0; j < N; j++)
        {
            float dis = efanna2e::pq_dist_lookup_single(&pq_data[j * num_pq_chunks], num_pq_chunks, dist_vec);
            res.emplace_back(dis, j);
        }
        printf("Collecting %u\n", i);
        std::sort(res.begin(), res.end());
        // 取出索引
        std::vector<unsigned> tmp;
        for (unsigned j = 0; j < 10; j++)
        {
            tmp.push_back(res[j].second);
        }
        res_indices.push_back(tmp);
    }

    unsigned gt_num, gt_dim;
    std::vector<std::vector<unsigned>> gt = common::read_gt(gt_file, "bin", gt_num, gt_dim);
    std::cout << "gt_num, gt_dim = " << gt_num << ", " << gt_dim << std::endl;

    float recall = common::compute_recall(qN, 10, gt, res_indices);
    printf("Recall: %.2lf\n", recall);
    // for (int i = 0; i < num_pq_chunks; i++){
    //     std::cout << "chunk " << i << ": ";
    //     for(int j = 0; j < NUM_PQ_CENTROIDS; j++){
    //         std::cout << dist_vec[i * NUM_PQ_CENTROIDS + j] << " ";
    //     }
    //     std::cout << std::endl << std::endl;
    // }
    // unsigned maxn = 0;
    // for(int i = N-10; i < N; i++){
    //     for(int j = 0; j < num_pq_chunks; j++){
    //         unsigned tmp = (unsigned)pq_data[i * num_pq_chunks + j];
    //         maxn = std::max(maxn, tmp);
    //         printf("%u ", tmp);
    //     }
    //     std::cout << std::endl << std::endl;
    // }
    // printf("max: %u\n", maxn);

    // for(int i = N-10; i<N; i++){
    //     float dis = efanna2e::pq_dist_lookup_single(&pq_data[i*num_pq_chunks], num_pq_chunks, dist_vec);
    //     printf("%f\n", dis);
    // }
    // delete[] dist_vec;
}