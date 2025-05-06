//
// Created by 付聪 on 2017/6/21.
// Modified  by 陈萌 on 2024/4/30
// 

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <xmmintrin.h>

#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
        // 4KB for unified files

#ifdef __APPLE__
#else
#include <malloc.h>
#endif
namespace efanna2e {

static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

inline float *data_align(float *data_ori, unsigned point_num, unsigned &dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif

    // std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
    float *data_new = 0;
    uint64_t pts = static_cast<uint64_t>(point_num);
    uint64_t d = static_cast<uint64_t>(dim);
    uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
// std::cout << "align to new dim: "<<new_dim << std::endl;
#ifdef __APPLE__
    data_new = new float[new_dim * point_num];
#else
    data_new = (float *)memalign(DATA_ALIGN_FACTOR * 8, pts * new_dim * sizeof(float));
#endif

    for (unsigned i = 0; i < pts; i++) {
        memcpy(data_new + i * new_dim, data_ori + i * d, d * sizeof(float));
        memset(data_new + i * new_dim + d, 0, (new_dim - d) * sizeof(float));
    }
    dim = new_dim;
    std::cout << "new_dim: " << dim << std::endl;
#ifdef __APPLE__
    delete[] data_ori;
#else
    // free(data_ori);
    delete[] data_ori;
#endif
    return data_new;
}

inline void prefetch_vector(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64) _mm_prefetch((const char *)vec + d, _MM_HINT_T0);
}

// load bin meta data from file with different data type, so use template
// get number of points and dimension
template <typename T>
void load_gt_meta(const char *filename, unsigned &points_num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&points_num, 4);
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint32_t calc_contained_pts = (unsigned)((fsize - sizeof(uint32_t) * 2) / (dim) / sizeof(T));
    std::cout << "[Loading] Load gt from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    if (points_num * 2 != calc_contained_pts) {
        std::cerr << "filename: " << std::string(filename) << std::endl;
        std::cerr << "Data file size wrong! Get points " << calc_contained_pts << " but should have " << points_num
                  << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}
template <typename T>
void load_meta(const char *filename, unsigned &points_num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&points_num, 4);
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint32_t calc_contained_pts = (unsigned)((fsize - sizeof(uint32_t) * 2) / (dim) / sizeof(T));
    std::cout << "[Loading] Load meta from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    if (points_num != calc_contained_pts) {
        std::cerr << "filename: " << std::string(filename) << std::endl;
        std::cerr << "Data file size wrong! Get points " << calc_contained_pts << " but should have " << points_num
                  << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T, typename T2>
void load_gt_data_with_dist(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data, T2 *&res_dists) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    data = new T[points_num * dim];
    res_dists = new T2[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * dim), dim * sizeof(T));
    }

    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(res_dists + i * dim), dim * sizeof(T2));
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != points_num * dim * sizeof(T) * 2 + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted!" << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T>
void load_gt_data(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    data = new T[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * dim), dim * sizeof(T));
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != points_num * dim * sizeof(T) + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted!" << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T>
void load_data(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data) {
    std::cout << "[Loading] Load data from file: " << filename;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    std::cout << " points_num: " << points_num << " dim: " << dim << std::endl;
    uint64_t pts = static_cast<uint64_t>(points_num);
    uint64_t d = static_cast<uint64_t>(dim);
    uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
    // data = new T[pts * d];
    // check T type

    data = (T*)memalign(DATA_ALIGN_FACTOR * 8, pts * new_dim * sizeof(T));
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * new_dim), d * sizeof(T));
        memset(data + i * new_dim + d, 0, (new_dim - d) * sizeof(T));
        // if ((i + 1) % 100000 == 0)
        //     std::cout << "i: " << i << std::endl;
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != pts * d * sizeof(T) + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted! filename:" << std::string(filename) << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    std::cout << "[Loading] Finish loading data from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    in.close();
}


template<typename T>
inline void normalize(T* arr, const size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    sum += arr[i] * arr[i];
  }
  sum = sqrt(sum);

  for (size_t i = 0; i < dim; i++) {
    arr[i] = (T)(arr[i] / sum);
  }
}
template<typename T>
inline void ip_normalize(T* arr, const size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    sum += arr[i] * arr[i];
  }
//   sum = sqrt(sum);

  for (size_t i = 0; i < dim; i++) {
    arr[i] = (T)(arr[i] / sum);
  }
}



template<typename InType, typename OutType>
void convert_types(const InType* srcmat, OutType* destmat, size_t npts, size_t dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t) npts; i++) {
        for (uint64_t j = 0; j < dim; j++) {
            destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
        }
    }
}
template<typename T>
inline size_t save_bin(const std::string& filename, T* data, size_t npts, size_t ndims, size_t offset = 0) {
    std::fstream writer(filename, std::ios::binary | std::ios::out);
    if (!writer.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    // open_file_to_write(writer, filename);

    std::cout << "Writing bin: " << filename.c_str() << std::endl;
    writer.seekp(offset, writer.beg);
    uint32_t    npts_u32 = (uint32_t) npts, ndims_u32 = (uint32_t) ndims;
    
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char*) &npts_u32, sizeof(uint32_t));
    writer.write((char*) &ndims_u32, sizeof(uint32_t));
    std::cout << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B" << std::endl;

    writer.write((char*) data, npts * ndims * sizeof(T));
    writer.close();
    std::cout << "Finished writing bin." << std::endl;
    return bytes_written;
}
template<typename T>
inline size_t save_bin_stream(std::fstream& writer, T* data, size_t npts, size_t ndims, size_t offset = 0) {
    if (!writer.is_open()) {
        throw std::runtime_error("cannot open file");
    }

    writer.seekp(offset, writer.beg);
    uint32_t npts_u32 = (uint32_t) npts, ndims_u32 = (uint32_t) ndims;
    
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char*) &npts_u32, sizeof(uint32_t));
    writer.write((char*) &ndims_u32, sizeof(uint32_t));
    std::cout << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B" << std::endl;

    writer.write((char*) data, npts * ndims * sizeof(T));
    return bytes_written;
}

// template<typename T>
// inline void load_bin_impl(std::basic_istream<char>& reader, std::unique_ptr<T[]> & data, size_t& npts, size_t& dim, size_t file_offset = 0) {
//     int npts_i32, dim_i32;

//     reader.seekg(file_offset, reader.beg);
//     reader.read((char*) &npts_i32, sizeof(int));
//     reader.read((char*) &dim_i32, sizeof(int));
//     npts = (unsigned) npts_i32;
//     dim = (unsigned) dim_i32;

//     std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

//     std::unique_ptr<float[]> data(new float[npts * dim]);
//     reader.read((char*) data, npts * dim * sizeof(T));
// }

template<typename T>
inline void load_bin_impl(std::basic_istream<char>& reader, T*& data, size_t& npts, size_t& dim, size_t file_offset = 0) {
    uint32_t npts_u32, dim_u32;
    reader.seekg(file_offset, reader.beg);
    std::cout << "11111111111111111111111111111111111111 " <<std::endl;
    reader.read((char*) &npts_u32, sizeof(uint32_t));
    std::cout << "11111111111111111111111111111111111111 " <<std::endl;
    reader.read((char*) &dim_u32, sizeof(uint32_t));
    std::cout << "222222222222222222222222222222222222222 " <<std::endl;
    npts = (size_t) npts_u32;
    dim = (size_t) dim_u32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

    data = new T[npts * dim];
    reader.read((char*) data, npts * dim * sizeof(T));
}


// template <typename T>
// void load_bin(const std::string& bin_file, std::unique_ptr<T[]> &data, size_t& npts, size_t& dim, size_t offset = 0) {
//     std::cout << "Reading bin file " << bin_file.c_str() << " ..." << std::endl;
//     std::ifstream reader;
//     reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

//     std::cout << "Opening bin file " << bin_file.c_str() << "... " << std::endl;
//     reader.open(bin_file, std::ios::binary | std::ios::ate);
//     reader.seekg(0);
//     load_bin_impl<T>(reader, data, npts, dim, offset);

//     std::cout << "done." << std::endl;
// }

template <typename T>
inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0)
{
    std::cout << "Reading bin file " << bin_file.c_str() << " ..." << std::endl;
    std::ifstream reader;
    // reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    std::cout << "Opening bin file " << bin_file.c_str() << "... " << std::endl;
    reader.open(bin_file, std::ios::binary);
    // reader.seekg(0);
    // load_bin_impl<T>(reader, data, npts, dim, offset);

    uint32_t npts_u32, dim_u32;
    reader.seekg(offset, std::ios::beg);

    reader.read((char*) &npts_u32, sizeof(uint32_t));
    reader.read((char*) &dim_u32, sizeof(uint32_t));
    npts = (size_t) npts_u32;
    dim = (size_t) dim_u32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

    data = new T[npts * dim];
    reader.read((char*) data, npts * dim * sizeof(T));

    std::cout << "done." << std::endl;
}


inline bool file_exists(const std::string& name, bool dirCheck = false) {
  int val;
  struct stat buffer;
  val = stat(name.c_str(), &buffer);

  if (val != 0) {
    switch (errno) {
      case EINVAL:
        std::cout << "Invalid argument passed to stat()" << std::endl;
        break;
      case ENOENT:
        // file is not existing, not an issue, so we won't cout anything.
        break;
      default:
        std::cout << "Unexpected error in stat():" << errno << std::endl;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}


#define IS_ALIGNED(value, alignment) (((value) % (alignment)) == 0)
inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr; 

    if (IS_ALIGNED(size, align) == 0) {
        fprintf(stderr, "Error: Requested size is not aligned to %zu\n", align);
        exit(EXIT_FAILURE);
    }

    *ptr = aligned_alloc(align, size);
    if (*ptr == nullptr) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

template<typename T>
inline void copy_aligned_data_from_file(const char* bin_file, T*& data,
                                        uint32_t& npts, uint32_t& dim,
                                        const uint32_t& rounded_dim,
                                        size_t        offset = 0) {
    if (data == nullptr) {
        std::cerr << "Memory was not allocated for " << data << " before calling the load function. Exiting..." << std::endl;
    }
    std::ifstream reader;
    reader.exceptions(std::ios::badbit | std::ios::failbit);
    reader.open(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    for (size_t i = 0; i < npts; i++) {
        reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
}


// Metric class to statistic time consuming
// class TimeMetric {
//    public:
//     TimeMetric() : start_(std::chrono::high_resolution_clock::now()) {}

//     void reset() { start_ = std::chrono::high_resolution_clock::now(); }

//     // return milliseconds
//     double elapsed() const {
//         auto end = std::chrono::high_resolution_clock::now();
//         return std::chrono::duration_cast<std::chrono::duration<double>>(end - start_).count();
//     }
//     // print accumulated time
//     void print(const std::string &prompt) { std::cout << prompt << ": " << elapsed_ << "ms" << std::endl; }
//     // accumulate elapsed time
//     void record() {
//         elapsed_ += elapsed();
//         // std::cout << prompt << ": " << elapsed_ << "s" << std::endl;
//         reset();
//     }

//    private:
//     std::chrono::high_resolution_clock::time_point start_;
//     // accumulate elapsed time
//     double elapsed_ = 0;
// };

}  // namespace efanna2e

#endif  // EFANNA2E_UTIL_H
