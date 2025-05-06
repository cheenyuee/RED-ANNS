#include <iostream>
#include <fstream>
#include <string>

#include "efanna2e/cached_io.h"
#include "math_utils.h"
// #include "utils.h"
// #include "partition.h"

#define SHOW_ENTRY_POINTS_INFO

#define ENTRY_BLOCK_SIZE 500000
// #define ENTRY_POINTS_DEBUG
namespace efanna2e {
    template<typename T>
    void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size, size_t &ndims) {
        size_t                          npts;
        uint32_t                        npts32, ndims32;
        std::vector<std::vector<float>> sampled_vectors;

        // amount to read in one shot
        uint64_t read_blk_size = 64 * 1024 * 1024;
        // create cached reader + writer
        cached_ifstream base_reader(data_file.c_str(), read_blk_size);

        // metadata: npts, ndims
        base_reader.read((char *) &npts32, sizeof(unsigned));
        base_reader.read((char *) &ndims32, sizeof(unsigned));
        npts = npts32;
        ndims = ndims32;

        std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
        p_val = p_val < 1 ? p_val : 1;

        std::random_device rd;  // Will be used to obtain a seed for the random number
        size_t             x = rd();
        std::mt19937       generator((unsigned) x);
        std::uniform_real_distribution<float> distribution(0, 1);

        for (size_t i = 0; i < npts; i++) {
            base_reader.read((char *) cur_vector_T.get(), ndims * sizeof(T));
            float rnd_val = distribution(generator);
            if (rnd_val < p_val) {
                std::vector<float> cur_vector_float;
                for (size_t d = 0; d < ndims; d++)
                    cur_vector_float.push_back(cur_vector_T[d]);
                sampled_vectors.push_back(cur_vector_float);
            }
        }
        slice_size = sampled_vectors.size();
        sampled_data = new float[slice_size * ndims];
        for (size_t i = 0; i < slice_size; i++) {
            for (size_t j = 0; j < ndims; j++) {
                sampled_data[i * ndims + j] = sampled_vectors[i][j];
            }
        }
    }

    template<typename T>
    // bool generate_more_entry_points(float *train_data_float, size_t num_train_data,
    //                                 size_t dim_train_data, size_t ep_num_parts,
    //                                 std::string data_file,
    //                                 const std::string index_prefix_path,
    //                                 size_t max_k_means_reps = 12) 
    bool generate_more_entry_points(const std::string data_file, size_t ep_num_parts,
                                    const std::string medoids_save_folder, double p_val = 0.3, size_t max_k_means_reps = 12) 
    {
        std::cout << "[generate entry points] Processing global k-means (kmeans_partitioning Step)" << std::endl;

        // Compute clusters&centroids based on sampled train_data
        std::cout << "[generate entry points] Sampling train data" << std::endl;
        size_t num_train_data, dim_train_data;
        float *train_data_float;
        // double p_val = 0.3;
        gen_random_slice<T>(data_file.c_str(), p_val, train_data_float, num_train_data, dim_train_data);
        
        // Select initial cluster centroids using k-means++ & Optimize cluster centroids using Lloyd's algorithm
        std::cout << "[generate entry points] clustering" << std::endl;
        float *pivot_data = new float[ep_num_parts * dim_train_data];
        kmeans::kmeanspp_selecting_pivots(train_data_float, num_train_data, dim_train_data, pivot_data, ep_num_parts);

        kmeans::run_lloyds(train_data_float, num_train_data, dim_train_data, pivot_data,
                           ep_num_parts, max_k_means_reps, NULL, NULL);



        // Scan every data to determine medoids 
        std::cout << "[generate entry points] Build medoids file" << std::endl;
        std::ifstream base_reader(data_file, std::ios::binary);
        uint32_t npts32;
        uint32_t basedim32;
        base_reader.read((char *) &npts32, sizeof(uint32_t));
        base_reader.read((char *) &basedim32, sizeof(uint32_t));

        size_t npts = npts32; 
        size_t dim = basedim32;
        size_t block_size = npts <= ENTRY_BLOCK_SIZE ? npts : ENTRY_BLOCK_SIZE;

        // Initialize containers for the nearest point to each centroid (ID and distance)
        std::vector<uint32_t> closest_id(ep_num_parts, 0);
        std::vector<float> closest_dis(ep_num_parts, 0x3fffffff); // Large initial value
        float *store_disk_data = new float[ep_num_parts * dim_train_data];
        memset(store_disk_data, 0, ep_num_parts * dim_train_data); // Initialize storage

        // Process each data block
        std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size);
        std::unique_ptr<float[]> block_closest_distance = std::make_unique<float[]>(block_size);
        std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
        std::unique_ptr<float[]> block_train_data_float = std::make_unique<float[]>(block_size * dim_train_data);
        
        size_t num_blocks = DIV_ROUND_UP(npts, block_size);
        for (size_t block = 0; block < num_blocks; block++) {
            size_t start_id = block * block_size;
            size_t end_id = (std::min)((block + 1) * block_size, npts);
            size_t cur_blk_size = end_id - start_id;

            // Read data block and convert it to float type
            base_reader.read((char *) block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
            efanna2e::convert_types<T, float>(block_data_T.get(), block_train_data_float.get(), cur_blk_size, dim);

            // Compute nearest centroid for each data point in the block
            math_utils::compute_closest_centers(block_train_data_float.get(), cur_blk_size, dim_train_data, 
                                                pivot_data, ep_num_parts, 1, 
                                                block_closest_centers.get(),
                                                block_closest_distance.get());

            // Update nearest point information for each centroid
            for (size_t p = 0; p < cur_blk_size; p++) {
                int centroids = block_closest_centers[p];
                int this_id = start_id + p;

                // check for medoids
                if (closest_dis[centroids] > block_closest_distance[p]) {
                    closest_dis[centroids] = block_closest_distance[p];
                    closest_id[centroids] = this_id;
                    memcpy(store_disk_data + centroids * dim, block_train_data_float.get() + p * dim, dim * sizeof(float));
                }
            }
        }

        // Save entry point data & IDs to disk
        std::string disk_entry_points_path = medoids_save_folder + "ncenter_" + std::to_string(ep_num_parts) + "_p_" + std::to_string(p_val) + "_plain.entry_points.bin";
        std::string disk_entry_points_id_path = medoids_save_folder + "ncenter_" + std::to_string(ep_num_parts) + "_p_" + std::to_string(p_val) + "_plain.entry_points_ids.bin";
        
        efanna2e::save_bin<float>(disk_entry_points_path.c_str(), store_disk_data, (size_t) ep_num_parts, dim);

        uint32_t *new_closest_id = new uint32_t[ep_num_parts];
        for (size_t i = 0; i < ep_num_parts; ++i) {
            new_closest_id[i] = closest_id[i];
        }
        efanna2e::save_bin<uint32_t>(disk_entry_points_id_path.c_str(), new_closest_id, (size_t) ep_num_parts, 1);

    #ifdef SHOW_ENTRY_POINTS_INFO
        for (size_t i = 0; i < ep_num_parts; ++i) {
            std::cout << new_closest_id[i] << " " << closest_dis[i] << std::endl;
        }
    #endif

        delete[] new_closest_id;
        delete[] store_disk_data;
        delete[] pivot_data;
        delete[] train_data_float;
        return true;
    }

}; 
