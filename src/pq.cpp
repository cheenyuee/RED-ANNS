#include <mkl.h>
#include "../include/pq.h"
#include "../include/math_utils.h"
#include "../include/efanna2e/util.h"
#include "../include/efanna2e/cached_io.h"
#include "../include/generate_subgraph.h"
// #include "partition.h"

#include "tsl/robin_map.h"

// block size for reading/processing large files and matrices in blocks
#define BLOCK_SIZE 3000000

namespace efanna2e {
  
// ####################################################################################################################
  FixedChunkPQTable::FixedChunkPQTable() {}

  FixedChunkPQTable::~FixedChunkPQTable() {
    if (tables != nullptr)
      delete[] tables;
    if (tables_tr != nullptr)
      delete[] tables_tr;
    if (chunk_offsets != nullptr)
      delete[] chunk_offsets;
    if (centroid != nullptr)
      delete[] centroid;
  }


  void FixedChunkPQTable::load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks) {
    uint64_t nr, nc;
    
    // Load meta data
    uint64_t* file_offset_data;
    size_t nr_size = static_cast<size_t>(nr);
    size_t nc_size = static_cast<size_t>(nc);
    load_bin<uint64_t>(pq_table_file, file_offset_data, nr_size, nc_size);

    // Check validation
    if (nr_size != 4) {
      std::cout << "Error reading pq_pivots file " << pq_table_file
                    << ". Offsets dont contain correct metadata, # offsets = "
                    << nr_size << ", but expecting " << 4 << std::endl;
    }
    else {
      std::cout << "Offsets: " << file_offset_data[0] << " "
                    << file_offset_data[1] << " " << file_offset_data[2] << " "
                    << file_offset_data[3] << std::endl;
    }

    // Load PQ_table: num_centers * dim
    efanna2e::load_bin<float>(pq_table_file, tables, nr, nc, file_offset_data[0]);
    if ((nr != NUM_PQ_CENTROIDS)) {
      std::cout << "Error reading pq_pivots file " << pq_table_file
                    << ". file_num_centers  = " << nr << " but expecting "
                    << NUM_PQ_CENTROIDS << " centers";
    }

    this->ndims = nc;

    // Load centroid: dim * 1
    efanna2e::load_bin<float>(pq_table_file, centroid, nr, nc, file_offset_data[1]);

    if ((nr != this->ndims) || (nc != 1)) {
      std::cerr << "Error reading centroids from pq_pivots file "
                    << pq_table_file << ". file_dim  = " << nr
                    << ", file_cols = " << nc << " but expecting "
                    << this->ndims << " entries in 1 dimension.";
    }

    int chunk_offsets_index = 2;

    // Load chunk_offsets: (chunk_size + 1) * 1
    efanna2e::load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);
    if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0)) {
      std::cerr << "Error loading chunk offsets file. numc: " << nc
                    << " (should be 1). numr: " << nr << " (should be "
                    << num_chunks + 1 << " or 0 if we need to infer)"
                    << std::endl;
    }

    this->n_chunks = nr - 1;
    std::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
                  << ", #dims: " << this->ndims
                  << ", #chunks: " << this->n_chunks << std::endl;


    // Alloc and compute transpose: tables ——> tables_tr
    tables_tr = new float[NUM_PQ_CENTROIDS * this->ndims];
    for (uint64_t i = 0; i < NUM_PQ_CENTROIDS; i++) {
      for (uint64_t j = 0; j < this->ndims; j++) {
        tables_tr[j * NUM_PQ_CENTROIDS + i] = tables[i * this->ndims + j];
      }
    }
  }



  uint32_t FixedChunkPQTable::get_num_chunks() {
    return static_cast<uint32_t>(n_chunks);
  }

  void FixedChunkPQTable::preprocess_query(float* query_vec) {
    for (uint32_t d = 0; d < ndims; d++) {
      query_vec[d] -= centroid[d];
    }
  }

  // Assumes pre-processed query. At each chunk, compute distances of q ——> centers
  void FixedChunkPQTable::populate_chunk_distances(const float* query_vec, float* dist_vec) {
    memset(dist_vec, 0, NUM_PQ_CENTROIDS * n_chunks * sizeof(float));  // n_chunks * 256
    
    // compute distances for each chunk
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (chunk * NUM_PQ_CENTROIDS);
      
      for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {  // for every sub_dimension
        const float* centers_dim_vec = tables_tr + (j * NUM_PQ_CENTROIDS);  // dim * 256(num_centers)
        for (uint64_t idx = 0; idx < NUM_PQ_CENTROIDS; idx++) {
          double diff = centers_dim_vec[idx] - (query_vec[j]);
          chunk_dists[idx] += (float) (diff * diff);
        }
      }
    }
  }

  // Assumes pre-processed query. At each chunk, compute IP_distances of q ——> centers
  void FixedChunkPQTable::populate_chunk_inner_products(const float* query_vec, float* dist_vec) {
    memset(dist_vec, 0, NUM_PQ_CENTROIDS * n_chunks * sizeof(float));  // n_chunks * 256
    
    // compute distances for each chunk
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (chunk * NUM_PQ_CENTROIDS);
      
      for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (NUM_PQ_CENTROIDS * j);
        for (uint64_t idx = 0; idx < NUM_PQ_CENTROIDS; idx++) {
          double prod = centers_dim_vec[idx] * query_vec[j];  // assumes that we are not shifting the vectors to
                                                              // mean zero, i.e., centroid array should be all zeros
          chunk_dists[idx] -= (float) prod;  
                             // returning negative to keep the search code clean
                             // (max inner product vs min distance)
        }
      }
    }
  }

  float FixedChunkPQTable::l2_distance(const float* query_vec, uint8_t* base_vec) {
    float res = 0;
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
      for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (NUM_PQ_CENTROIDS * j);
        float        diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
        res += diff * diff;
      }
    }
    return res;
  }

  float FixedChunkPQTable::inner_product(const float* query_vec, uint8_t* base_vec) {
    float res = 0;
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
      for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (NUM_PQ_CENTROIDS * j);
        float        diff = centers_dim_vec[base_vec[chunk]] * query_vec[j];  // assumes centroid is 0 to
                                                                              // prevent translation errors
        res += diff;
      }
    }
    return -res;  // returns negative value to simulate distances (max -> min conversion)
  }

  // Assumes no rotation is involved
  void FixedChunkPQTable::inflate_vector(uint8_t* base_vec, float* out_vec) {
    for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
      for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (NUM_PQ_CENTROIDS * j);
        out_vec[j] = centers_dim_vec[base_vec[chunk]] + centroid[j];
      }
    }
  }



// ####################################################################################################################
  float pq_dist_lookup_single(const uint8_t* pq_id, const uint64_t pq_nchunks, const float* pq_dists) {
      float dist_out = 0.0f;

      for (uint64_t chunk = 0; chunk < pq_nchunks; chunk++) {
          const float* chunk_dists = pq_dists + NUM_PQ_CENTROIDS * chunk;

          if (chunk < pq_nchunks - 1) {
              _mm_prefetch((char*) (chunk_dists + NUM_PQ_CENTROIDS), _MM_HINT_T0);
          }
          uint8_t pq_centerid = pq_id[chunk];
          dist_out += chunk_dists[pq_centerid];
      }

      return dist_out;
  }
  
  
  /* all_coords: [num_points * num_chunks]    
   * ndims--num_chunks
   * out: [num_ids * num_chunks]
   */
  void aggregate_coords(const std::vector<unsigned>& ids, const uint8_t* all_coords, const uint64_t ndims, uint8_t* out) {
    for (uint64_t i = 0; i < ids.size(); i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(uint8_t));
    }
  }

  /* pq_ids: [n_ids * num_chunks]
   * pq_dists: [num_chunks * 256]
   * dists_out: [n_ids]
   */ 
  void pq_dist_lookup(const uint8_t* pq_ids, const uint64_t n_pts, const uint64_t pq_nchunks, const float* pq_dists, std::vector<float> &dists_out) {
    //_mm_prefetch((char*) dists_out, _MM_HINT_T0);
    _mm_prefetch((char*) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char*) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char*) (pq_ids + 128), _MM_HINT_T0);
    dists_out.clear();
    dists_out.resize(n_pts, 0);
    for (uint64_t chunk = 0; chunk < pq_nchunks; chunk++) {
      const float* chunk_dists = pq_dists + chunk * 256;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char*) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (uint64_t idx = 0; idx < n_pts; idx++) {
        uint8_t pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }

  // Need to replace calls to these functions with calls to vector & based functions above
  void aggregate_coords(const unsigned* ids, const uint64_t n_ids, const uint8_t* all_coords, const uint64_t ndims, uint8_t* out) {
    for (uint64_t i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(uint8_t));
    }
  }

  void pq_dist_lookup(const uint8_t* pq_ids, const uint64_t n_pts, const uint64_t pq_nchunks, const float* pq_dists, float* dists_out) {
    _mm_prefetch((char*) dists_out, _MM_HINT_T0);
    _mm_prefetch((char*) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char*) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char*) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (uint64_t chunk = 0; chunk < pq_nchunks; chunk++) {
      const float* chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char*) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (uint64_t idx = 0; idx < n_pts; idx++) {
        uint8_t pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }




// ####################################################################################################################
  // Given training data in train_data of dimensions num_train * dim, generate PQ pivots using k-means algorithm to 
  // partition the co-ordinates into num_pq_chunks (if it divides dimension, else rounded) chunks, and runs k-means 
  // in each chunk to compute the PQ pivots and stores in bin format in file pq_pivots_path as num_centers*dim floating point binary file
  int generate_pq_pivots(const float* const passed_train_data, size_t num_train, unsigned dim, 
                         unsigned num_centers, unsigned num_pq_chunks, 
                         unsigned max_k_means_reps, std::string pq_pivots_path, bool make_zero_mean) 
  {
    if (num_pq_chunks > dim) {
      std::cout << " Error: number of chunks more than dimension" << std::endl;
      return -1;
    }

    std::unique_ptr<float[]> train_data = std::make_unique<float[]>(num_train * dim);
    std::memcpy(train_data.get(), passed_train_data, num_train * dim * sizeof(float));

    float *full_pivot_data = new float[num_centers * dim];
    memset(full_pivot_data, 0, num_centers * dim); // Initialize storage


    // If pq_pivots of this shape already exist, we don't need to do it again
    if (file_exists(pq_pivots_path)) {
      size_t file_dim, file_num_centers;
      efanna2e::load_bin<float>(pq_pivots_path, full_pivot_data, file_num_centers, file_dim, METADATA_SIZE);
      if (file_dim == dim && file_num_centers == num_centers) {
        std::cout << "PQ pivot file exists. Not generating again" << std::endl;
        return -1;
      }
    }


    // Calculate centroid and center the training data
    std::unique_ptr<float[]> centroid = std::make_unique<float[]>(dim);
    for (size_t d = 0; d < dim; d++) centroid[d] = 0;
    
    if (make_zero_mean) {  // If we use L2 distance, there is an option to translate all vectors to  
                           // make them centered and then compute PQ. This needs to be set to false  
                           // when using PQ for MIPS as such translations dont preserve inner products.
      for (size_t d = 0; d < dim; d++) {
        for (size_t p = 0; p < num_train; p++) 
          centroid[d] += train_data[p * dim + d];
        
        centroid[d] /= num_train;
      }

      for (size_t d = 0; d < dim; d++) {
        for (size_t p = 0; p < num_train; p++) {
          train_data[p * dim + d] -= centroid[d];
        }
      }
    }
    
    

    // Assign each dimension to a single chunk
    size_t low_val = (size_t) std::floor((double) dim / (double) num_pq_chunks);
    size_t high_val = (size_t) std::ceil((double) dim / (double) num_pq_chunks);
    size_t max_num_high = dim - (low_val * num_pq_chunks);
    size_t cur_num_high = 0;
    size_t cur_bin_threshold = high_val;  // start with high_val
   
    std::vector<std::vector<uint32_t>> bin_to_dims(num_pq_chunks);  // num_pq_chunks x [low_val, high_val]
    tsl::robin_map<uint32_t, uint32_t> dim_to_bin;
    std::vector<float>                 bin_loads(num_pq_chunks, 0);

    for (uint32_t d = 0; d < dim; d++) {
      if (dim_to_bin.find(d) != dim_to_bin.end())
        continue;
      
      // search for tiniest bin which also satisfy cur_bin_threshold
      auto  cur_best = num_pq_chunks + 1;
      float cur_best_load = std::numeric_limits<float>::max();
      for (uint32_t b = 0; b < num_pq_chunks; b++) {
        if (bin_loads[b] < cur_best_load && bin_to_dims[b].size() < cur_bin_threshold) {
          cur_best = b;
          cur_best_load = bin_loads[b];
        }
      }
      bin_to_dims[cur_best].push_back(d);

      // update cur_num_high ——> max_num_high
      if (bin_to_dims[cur_best].size() == high_val) {
        cur_num_high++;
        if (cur_num_high == max_num_high)
          cur_bin_threshold = low_val;
      }
    }
    // Build chunk_offsets array: chunk_offsets[i] is where chunk_i starts
    std::vector<uint32_t> chunk_offsets;
    chunk_offsets.clear();
    chunk_offsets.push_back(0);

    for (uint32_t b = 0; b < num_pq_chunks; b++) {
      if (b > 0)
        chunk_offsets.push_back(chunk_offsets[b - 1] + (uint32_t) bin_to_dims[b - 1].size());
    }
    chunk_offsets.push_back(dim);




    // Calculate centers[num_centers * cur_chunk_size] for every chunk
    // full_pivot_data.reset(new float[num_centers * dim]);
    // delete[] full_pivot_data;
    // full_pivot_data = new float[num_centers * dim];

    for (size_t i = 0; i < num_pq_chunks; i++) {
      size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];  // dim for this subspace

      if (cur_chunk_size == 0)
        continue;
      std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
      std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_train * cur_chunk_size);
      std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_train);

      std::cout << "Processing chunk " << i << " with dimensions [" << chunk_offsets[i] << ", " << chunk_offsets[i + 1] << ")" << std::endl;

#pragma omp parallel for schedule(static, 65536)
      for (int64_t j = 0; j < (int64_t) num_train; j++) {
        std::memcpy(cur_data.get() + j * cur_chunk_size, 
                    train_data.get() + j * dim + chunk_offsets[i], 
                    cur_chunk_size * sizeof(float));
      }

      kmeans::kmeanspp_selecting_pivots(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers);
      kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size,
                         cur_pivot_data.get(), num_centers, max_k_means_reps,
                         NULL, closest_center.get());

      for (uint64_t j = 0; j < num_centers; j++) {
        std::memcpy(full_pivot_data + j * dim + chunk_offsets[i], cur_pivot_data.get() + j * cur_chunk_size, cur_chunk_size * sizeof(float));
      }
    }



    // Save file: METADATA_SIZE + 
    // std::string pq_pivots_path_a = pq_pivots_path + std::string("a");
    // std::string pq_pivots_path_b = pq_pivots_path + std::string("b");
    // std::string pq_pivots_path_c = pq_pivots_path + std::string("c");

    // save_bin<float>(pq_pivots_path_a.c_str(), full_pivot_data, (size_t) num_centers, dim);
    // save_bin<float>(pq_pivots_path_b.c_str(), centroid.get(), (size_t) dim, 1);
    // save_bin<uint32_t>(pq_pivots_path_c.c_str(), chunk_offsets.data(), chunk_offsets.size(), 1);

    std::fstream writer(pq_pivots_path.c_str(), std::ios::binary | std::ios::out);
    if (!writer.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    std::vector<size_t> cumul_bytes(4, 0);
    cumul_bytes[0] = METADATA_SIZE;

    cumul_bytes[1] = cumul_bytes[0] + save_bin_stream<float>(writer, full_pivot_data,
                                                      (size_t) num_centers, dim, cumul_bytes[0]);
    cumul_bytes[2] = cumul_bytes[1] + save_bin_stream<float>(writer, centroid.get(), 
                                                      (size_t) dim, 1, cumul_bytes[1]);
    cumul_bytes[3] = cumul_bytes[2] + save_bin_stream<uint32_t>(writer, chunk_offsets.data(),
                                                      chunk_offsets.size(), 1, cumul_bytes[2]);
    save_bin_stream<size_t>(writer, cumul_bytes.data(), cumul_bytes.size(), 1, 0);

    std::cout << "Saved pq pivot data to " << pq_pivots_path << " of size " << cumul_bytes[cumul_bytes.size() - 1] << "B." << std::endl;
    writer.close();

    return 0;
  }


  // Streams the base file (data_file), and computes the closest centers in each chunk to 
  // generate the compressed data_file and stores it in pq_compressed_vectors_path.
  // If the number of centers is < 256, it stores as byte vector, else as 4-byte vector in binary format.
  template<typename T>
  int generate_pq_data_from_pivots(const std::string data_file,
                                   unsigned num_centers, unsigned num_pq_chunks,
                                   std::string pq_pivots_path,
                                   std::string pq_compressed_vectors_path) 
  {
    uint64_t            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t            npts32;
    uint32_t            basedim32;
    base_reader.read((char*) &npts32, sizeof(uint32_t));
    base_reader.read((char*) &basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    size_t dim = basedim32;

    float*    full_pivot_data;
    float*    centroid;
    uint32_t* chunk_offsets;

    // Load PQ pivot information
    if (!file_exists(pq_pivots_path)) {
      std::cout << "ERROR: PQ k-means pivot file not found" << std::endl;
    } 
    else {
      uint64_t nr, nc;
      uint64_t* file_offset_data;

      efanna2e::load_bin<uint64_t>(pq_pivots_path.c_str(), file_offset_data, nr, nc, 0);
      if (nr != 4) {
        std::cout << "Error reading pq_pivots file " << pq_pivots_path
                      << ". Offsets dont contain correct metadata, # offsets = "
                      << nr << ", but expecting 4.";
      }

      efanna2e::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data, nr, nc, file_offset_data[0]);  // num_centers * dim
      if ((nr != num_centers) || (nc != dim)) {
        std::cout << "Error reading pq_pivots file " << pq_pivots_path
                      << ". file_num_centers  = " << nr << ", file_dim = " << nc
                      << " but expecting " << num_centers << " centers in " << dim << " dimensions." << std::endl;
      }


      efanna2e::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc, file_offset_data[1]);  // dim * 1
      if ((nr != dim) || (nc != 1)) {
        std::cout << "Error reading pq_pivots file " << pq_pivots_path
                      << ". file_dim = " << nr << ", file_cols = " << nc
                      << " but expecting " << dim << " entries in 1 dimension." << std::endl;
      }

      efanna2e::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc, file_offset_data[2]);  // num_pq_chunks * 1
      if (nr != (uint64_t) num_pq_chunks + 1 || nc != 1) {
        std::cout << "Error reading pq_pivots file at chunk offsets; file has nr="
                      << nr << ",nc=" << nc << ", expecting nr=" << num_pq_chunks + 1
                      << ", nc=1." << std::endl;
      }

      std::cout << "Loaded PQ pivot information" << std::endl;
    }

    // Compress base_data into [num_points * num_pq_chunks]
    std::ofstream compressed_file_writer(pq_compressed_vectors_path, std::ios::binary);
    uint32_t num_pq_chunksuint32_t = num_pq_chunks;

    compressed_file_writer.write((char*) &num_points, sizeof(uint32_t));
    compressed_file_writer.write((char*) &num_pq_chunksuint32_t, sizeof(uint32_t));

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;


    std::unique_ptr<uint32_t[]> block_compressed_base = std::make_unique<uint32_t[]>(block_size * (uint64_t) num_pq_chunks);
    std::memset(block_compressed_base.get(), 0, block_size * (uint64_t) num_pq_chunks * sizeof(uint32_t));

    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);
 
    for (size_t block = 0; block < num_blocks; block++) {
      size_t start_id = block * block_size;
      size_t end_id = (std::min)((block + 1) * block_size, num_points);
      size_t cur_blk_size = end_id - start_id;

      base_reader.read((char*) (block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
      efanna2e::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

      std::cout << "Processing points  [" << start_id << ", " << end_id << ").." << std::flush;

      // Centerize
      for (uint64_t p = 0; p < cur_blk_size; p++) {
        for (uint64_t d = 0; d < dim; d++) {
          block_data_tmp[p * dim + d] -= centroid[d];
        }
      }
      for (uint64_t p = 0; p < cur_blk_size; p++) {
        for (uint64_t d = 0; d < dim; d++) {
          block_data_float[p * dim + d] = block_data_tmp[p * dim + d];
        }
      }

      // At each chunk, compute closest center for block[0 - block_size]
      for (size_t i = 0; i < num_pq_chunks; i++) {
        size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
        if (cur_chunk_size == 0)
          continue;

        std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
        std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(cur_blk_size * cur_chunk_size);
        std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(cur_blk_size);

#pragma omp parallel for schedule(static, 8192)  // get current data [block_size * cur_chunk_size]
        for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
          for (uint64_t k = 0; k < cur_chunk_size; k++)
            cur_data[j * cur_chunk_size + k] = block_data_float[j * dim + chunk_offsets[i] + k];
        }

#pragma omp parallel for schedule(static, 1) // get current pivots [num_centers * cur_chunk_size]
        for (int64_t j = 0; j < (int64_t) num_centers; j++) {
          std::memcpy(cur_pivot_data.get() + j * cur_chunk_size,
                      full_pivot_data + j * dim + chunk_offsets[i],
                      cur_chunk_size * sizeof(float));
        }

        math_utils::compute_closest_centers(cur_data.get(), cur_blk_size, cur_chunk_size, 
                                            cur_pivot_data.get(), num_centers, 
                                            1, closest_center.get());

#pragma omp parallel for schedule(static, 8192)  // for every item, update it's closest_center
        for (int64_t j = 0; j < (int64_t) cur_blk_size; j++) {
          block_compressed_base[j * num_pq_chunks + i] = closest_center[j];  
        }
      }

      // Write [cur_blk_size * num_pq_chunks] to file
      if (num_centers > 256) {
        compressed_file_writer.write((char*) (block_compressed_base.get()), cur_blk_size * num_pq_chunks * sizeof(uint32_t));
      } 
      else {
        std::unique_ptr<uint8_t[]> pVec = std::make_unique<uint8_t[]>(cur_blk_size * num_pq_chunks);
        efanna2e::convert_types<uint32_t, uint8_t>(block_compressed_base.get(), pVec.get(), cur_blk_size, num_pq_chunks);
        compressed_file_writer.write((char*) (pVec.get()), cur_blk_size * num_pq_chunks * sizeof(uint8_t));
      }
      std::cout << ".done." << std::endl;
    }

    compressed_file_writer.close();
    return 0;
  }


  template<typename T>
  void generate_quantized_data(const std::string data_file_to_use,
                               const std::string pq_pivots_path,
                               const std::string pq_compressed_vectors_path,
                               efanna2e::Metric   compareMetric,
                               const double p_val, const size_t num_pq_chunks) 
  {
    // Instantiates train_data with random sample updates train_size
    size_t train_size, train_dim;
    float* train_data;

    gen_random_slice<T>(data_file_to_use.c_str(), p_val, train_data, train_size, train_dim);
    std::cout << "Training data with " << train_size << " samples loaded." << std::endl;

    bool make_zero_mean = false;
    if (compareMetric == efanna2e::Metric::INNER_PRODUCT)
      make_zero_mean = false;

    generate_pq_pivots(train_data, train_size, (uint32_t) train_dim,
                       NUM_PQ_CENTROIDS, (uint32_t) num_pq_chunks,
                       NUM_KMEANS_REPS_PQ, pq_pivots_path, make_zero_mean);
    std::cout << "#########################################################################################" << std::endl;
    generate_pq_data_from_pivots<T>(data_file_to_use.c_str(), 
                                    NUM_PQ_CENTROIDS, (uint32_t) num_pq_chunks, 
                                    pq_pivots_path,
                                    pq_compressed_vectors_path);

    delete[] train_data;
  }



  // Instantations of supported templates
  template int generate_pq_data_from_pivots<int8_t>(
      const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
      std::string pq_pivots_path, std::string pq_compressed_vectors_path);
  template int generate_pq_data_from_pivots<uint8_t>(
      const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
      std::string pq_pivots_path, std::string pq_compressed_vectors_path);
  template int generate_pq_data_from_pivots<float>(
      const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
      std::string pq_pivots_path, std::string pq_compressed_vectors_path);

  template void generate_quantized_data<int8_t>(
      const std::string data_file_to_use, const std::string pq_pivots_path,
      const std::string pq_compressed_vectors_path,
      efanna2e::Metric compareMetric, const double p_val,
      const size_t num_pq_chunks);

  template void generate_quantized_data<uint8_t>(
      const std::string data_file_to_use, const std::string pq_pivots_path,
      const std::string pq_compressed_vectors_path,
      efanna2e::Metric compareMetric, const double p_val,
      const size_t num_pq_chunks);

  template void generate_quantized_data<float>(
      const std::string data_file_to_use, const std::string pq_pivots_path,
      const std::string pq_compressed_vectors_path,
      efanna2e::Metric compareMetric, const double p_val,
      const size_t num_pq_chunks);
}  // namespace efanna2e
