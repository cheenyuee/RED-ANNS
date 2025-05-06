#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <iomanip>
#include <cstring>
#include <unordered_map>

/*
 *  查询统计相关
 */
namespace common
{

    class Timer
    {
        typedef std::chrono::high_resolution_clock _clock;
        std::chrono::time_point<_clock> check_point;

    public:
        Timer() : check_point(_clock::now())
        {
        }

        void reset()
        {
            check_point = _clock::now();
        }

        long long elapsed() const
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(_clock::now() - check_point).count();
        }

        float elapsed_seconds() const
        {
            return (float)elapsed() / 1000000.0f;
        }

        std::string elapsed_seconds_for_step(const std::string &step) const
        {
            return std::string("Time for ") + step + std::string(": ") + std::to_string(elapsed_seconds()) +
                   std::string(" seconds");
        }
    };

    struct QueryStats
    {
        unsigned query_index = 0;

        // NSG 相关统计量
        float total_us = 0.0; // total time to process query in micros
        unsigned n_hops = 0;  // search hops
        unsigned n_cmps = 0;  // cmps

        // EPG 相关统计量
        unsigned _EPG_n_hops = 0;
        unsigned _EPG_n_cmps = 0;

        // NUMA 相关统计量
        int should_run_on_node = -1;
        unsigned n_hops_local = 0;
        unsigned n_hops_remote = 0;
        unsigned n_cmps_local = 0;
        unsigned n_cmps_remote = 0;
        unsigned n_cmps_remote_cached = 0;
        unsigned n_res_local = 0;
        unsigned n_res_remote = 0;

        // 1-NN 相关统计量
        unsigned n_hops_1NN_chekpoint = 0;
        unsigned n_cmps_1NN_chekpoint = 0;
        unsigned n_cmps_local_1NN_chekpoint = 0;
        unsigned n_cmps_remote_1NN_chekpoint = 0;
    };

    template <typename T>
    inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
    {
        double avg = 0;
        for (uint64_t i = 0; i < len; i++)
        {
            avg += (double)member_fn(stats[i]);
        }
        return avg / len;
    }

    template <typename T>
    inline uint64_t get_total_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
    {
        uint64_t total = 0;
        for (uint64_t i = 0; i < len; i++)
        {
            total += (uint64_t)member_fn(stats[i]);
        }
        return total;
    }

    inline void print_query_stats(unsigned query_num, QueryStats *stats)
    {
        if (stats == nullptr)
            throw std::runtime_error("error@print_query_stats");
        // auto mean_latency = get_mean_stats<float>(stats, query_num,
        //                                           [](const common::QueryStats &stats)
        //                                           { return stats.total_us; });
        auto mean_hops = get_mean_stats<unsigned>(stats, query_num,
                                                  [](const common::QueryStats &stats)
                                                  { return stats.n_hops; });
        auto mean_cmps = get_mean_stats<unsigned>(stats, query_num,
                                                  [](const common::QueryStats &stats)
                                                  { return stats.n_cmps; });
        // auto mean_cmps_local = get_mean_stats<unsigned>(stats, query_num,
        //                                                 [](const QueryStats &stats)
        //                                                 { return stats.n_cmps_local; });
        // auto mean_cmps_remote = get_mean_stats<unsigned>(stats, query_num,
        //                                                  [](const QueryStats &stats)
        //                                                  { return stats.n_cmps_remote; });
        auto total_cmps = get_total_stats<unsigned>(stats, query_num,
                                                    [](const common::QueryStats &stats)
                                                    { return stats.n_cmps; });
        std::cout << std::fixed << std::setprecision(2);
        // std::cout << "mean_latency: " << mean_latency << std::endl;
        std::cout << "mean_hops: " << mean_hops << std::endl;
        std::cout << "mean_cmps: " << mean_cmps << std::endl;
        // std::cout << "mean_cmps_local: " << mean_cmps_local << "(" << ((float)mean_cmps_local / mean_cmps) * 100 << "%)" << std::endl;
        // std::cout << "mean_cmps_remote: " << mean_cmps_remote << "(" << ((float)mean_cmps_remote / mean_cmps) * 100 << "%)" << std::endl;
        std::cout << "total_cmps: " << total_cmps << std::endl;

        // ration of queries on node_1(when only 2 node)
        // auto total_should_run_on_node = get_total_stats<int>(stats, query_num,
        //                                                      [](const QueryStats &stats)
        //                                                      { return stats.should_run_on_node; });
        // std::cout << "total_should_run_on_node: " << total_should_run_on_node << "(" << ((float)total_should_run_on_node / query_num) * 100 << "%)" << std::endl;

        // auto EPG_mean_hops = get_mean_stats<unsigned>(stats, query_num,
        //                                               [](const common::QueryStats &stats)
        //                                               { return stats._EPG_n_hops; });
        // auto EPG_mean_cmps = get_mean_stats<unsigned>(stats, query_num,
        //                                               [](const common::QueryStats &stats)
        //                                               { return stats._EPG_n_cmps; });
        // std::cout << "EPG_mean_hops: " << EPG_mean_hops << std::endl;
        // std::cout << "EPG_mean_cmps: " << EPG_mean_cmps << std::endl;
    }

    inline constexpr int64_t do_align(int64_t x, int64_t align)
    {
        return (x + align - 1) / align * align;
    }

    inline constexpr size_t upper_div(size_t x, size_t y)
    {
        return (x + y - 1) / y; // upper[x / y]
    }

    // 求众数
    template <typename T>
    inline T find_mode(const std::vector<T> &nums)
    {
        if (nums.empty())
            throw std::runtime_error("error@find_mode: nums is empty.");
        std::unordered_map<T, uint64_t> frequency;
        T mode = nums[0];
        uint64_t maxCount = 1;
        for (T num : nums)
        {
            if (frequency.find(num) == frequency.end())
                frequency[num] = 1;
            else
                frequency[num]++;
            if (frequency[num] > maxCount)
            {
                maxCount = frequency[num];
                mode = num;
            }
        }
        return mode;
    }
} // namespace numaann

/*
 *  数据读写相关
 */
namespace common
{
    inline void normalize_data(float *data, unsigned N, unsigned Dim)
    {
        for (size_t i = 0; i < (size_t)N; i++)
        {
            float *point = data + (uint64_t)i * Dim;
            double norm = 0.0;
            for (size_t d = 0; d < (size_t)Dim; d++)
                norm += ((double)point[d] * point[d]);
            norm = sqrt(norm);
            if (norm == 0.0)
                throw std::runtime_error("error@normalize_data: norm can't be 0.0");
            for (size_t d = 0; d < (size_t)Dim; d++)
                point[d] /= norm;
        }
    }

    template <typename T>
    inline T *sample_data(const T *data, unsigned N, unsigned Dim, unsigned sample_num)
    {
        if (sample_num > N)
            throw std::runtime_error("error@sample_data: sample_num > N.");

        std::vector<unsigned> sampled_id;
        std::vector<bool> visited(N, false);
        unsigned sample_count = 0;
        while (sample_count < sample_num)
        {
            unsigned id = rand() % N;
            if (visited[id])
                continue;
            sampled_id.push_back(id);
            visited[id] = true;
            sample_count++;
        }

        T *sampled_data = new T[(uint64_t)sample_num * Dim];
        if (sampled_data == nullptr)
            throw std::runtime_error("error@sample_data: sampled_data is nullptr.");

        T *destination = sampled_data;
        for (size_t i = 0; i < sampled_id.size(); i++)
        {
            std::memcpy(destination, data + (uint64_t)sampled_id[i] * Dim, Dim * sizeof(T));
            destination += Dim;
        }
        return sampled_data;
    }

    template <typename T>
    inline float get_recall(std::vector<T> r1, std::vector<T> r2, unsigned K)
    {
        if (r1.size() < K or r2.size() < K)
            throw std::runtime_error("error@get_recall: groundtruth or reuslt is not enough.");
        std::set<T> a(r1.begin(), r1.begin() + K);
        std::set<T> b(r2.begin(), r2.begin() + K);
        std::set<T> result;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), inserter(result, result.begin()));
        return static_cast<float>(result.size()) / K;
    }

    template <typename T>
    inline float compute_recall(unsigned query_num, unsigned K, const std::vector<std::vector<T>> &gt, const std::vector<std::vector<T>> &res)
    {
        if (query_num > gt.size() or query_num > res.size())
            throw std::runtime_error("error@compute_recall");
        float recall = 0.0;
        for (size_t i = 0; i < (size_t)query_num; i++)
        {
            recall += get_recall(gt[i], res[i], K);
        }
        // printf("Recall@%d: %.2lf\n", K, recall * 100 / query_num);
        return recall * 100 / query_num;
    }

    inline long int fsize(FILE *fp)
    {
        long int prev = ftell(fp);
        fseek(fp, 0L, SEEK_END);
        long int sz = ftell(fp);
        fseek(fp, prev, SEEK_SET);
        return sz;
    }

    // 读 .vecs 格式 base、query 数据集，如 .fvecs 和 .bvecs
    template <typename T>
    inline T *read_data_vecs(const std::string &path_file, unsigned &N, unsigned &Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "rb");
        if (F == NULL)
            throw std::runtime_error("error@read_data_vecs: Dataset not found");

        [[maybe_unused]] size_t xxx = fread(&Dim, sizeof(unsigned), 1, F);
        long int sizebytes = fsize(F);
        N = sizebytes / (sizeof(unsigned) + sizeof(T) * Dim);
        rewind(F);
        T *data = new T[(uint64_t)N * Dim];
        for (size_t i = 0; i < (size_t)N; i++)
        {
            xxx = fread(&Dim, sizeof(unsigned), 1, F);
            xxx = fread(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
        }
        fclose(F);
        return data;
    }

    // 读 .bin 格式 base、query 数据集，如 .fbin 和 .bbin
    template <typename T>
    inline T *read_data_bin(const std::string &path_file, unsigned &N, unsigned &Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "rb");
        if (F == NULL)
            throw std::runtime_error("error@read_data_bin: Dataset not found");

        [[maybe_unused]] size_t xxx = fread(&N, sizeof(unsigned), 1, F);
        xxx = fread(&Dim, sizeof(unsigned), 1, F);
        T *data = new T[(uint64_t)N * Dim];
        // for (size_t i = 0; i < (size_t)N; i++)
        // {
        //     xxx = fread(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
        // }
        xxx = fread(data, sizeof(T), (uint64_t)N * Dim, F);
        fclose(F);
        return data;
    }

    // 读 .ivecs 格式 gt 数据集
    template <typename T>
    inline std::vector<std::vector<T>> read_gt_ivecs(const std::string &path_file, unsigned &N, unsigned &Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "rb");
        if (F == NULL)
            throw std::runtime_error("error@read_gt_ivecs: Dataset not found");

        [[maybe_unused]] size_t xxx = fread(&Dim, sizeof(unsigned), 1, F);
        long int sizebytes = fsize(F);
        N = sizebytes / (sizeof(unsigned) + sizeof(T) * Dim);
        rewind(F);
        std::vector<std::vector<T>> gt;
        for (size_t i = 0; i < (size_t)N; i++)
        {
            xxx = fread(&Dim, sizeof(unsigned), 1, F);
            T *nn = new T[Dim];
            xxx = fread(nn, sizeof(T), Dim, F);
            gt.push_back(std::vector<T>(nn, nn + Dim));
            delete[] nn;
        }
        fclose(F);
        return gt;
    }

    // 读 .ibin 格式 gt 数据集
    template <typename T>
    inline std::vector<std::vector<T>> read_gt_ibin(const std::string &path_file, unsigned &N, unsigned &Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "rb");
        if (F == NULL)
            throw std::runtime_error("error@read_gt_ibin: Dataset not found");

        [[maybe_unused]] size_t xxx = fread(&N, sizeof(unsigned), 1, F);
        xxx = fread(&Dim, sizeof(unsigned), 1, F);
        std::vector<std::vector<T>> gt;
        for (size_t i = 0; i < (size_t)N; i++)
        {
            T *nn = new T[Dim];
            xxx = fread(nn, sizeof(T), Dim, F);
            gt.push_back(std::vector<T>(nn, nn + Dim));
            delete[] nn;
        }
        fclose(F);
        return gt;
    }

    // 写 .vecs 格式 base、query 数据集，如 .fvecs 和 .bvecs
    template <typename T>
    inline void write_data_vecs(const std::string &path_file, T *data, unsigned N, unsigned Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "wb");
        if (F == NULL)
            throw std::runtime_error("error@write_data_vecs: File not found");

        [[maybe_unused]] size_t xxx;
        for (size_t i = 0; i < (size_t)N; i++)
        {
            xxx = fwrite(&Dim, sizeof(unsigned), 1, F);
            xxx = fwrite(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
        }
        fclose(F);
    }

    template <typename T>
    inline void write_data_bin(const std::string &path_file, T *data, unsigned N, unsigned Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "wb");
        if (F == NULL)
            throw std::runtime_error("error@write_data_bin: File not found");

        [[maybe_unused]] size_t xxx;
        xxx = fwrite(&N, sizeof(unsigned), 1, F);
        xxx = fwrite(&Dim, sizeof(unsigned), 1, F);
        for (size_t i = 0; i < (size_t)N; i++)
        {
            xxx = fwrite(data + (uint64_t)i * Dim, sizeof(T), Dim, F);
        }
        fclose(F);
    }

    // 明文形式保存 float 类型 base 数据集
    inline void write_data_txt(const std::string &path_file, float *base, unsigned N, unsigned Dim)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "w");
        if (F == NULL)
            throw std::runtime_error("error@write_base_txt: File not found");

        for (size_t i = 0; i < (size_t)N; i++)
        {
            for (size_t j = 0; j < (size_t)Dim; j++)
            {
                fprintf(F, "\t%f", *(base + (uint64_t)i * Dim + j));
            }
            fprintf(F, "\n");
        }
        fclose(F);
    }

    // 写 .vecs 格式 gt 数据集，如 .ivecs
    template <typename T>
    inline void write_gt_ivecs(const std::string &path_file, const std::vector<std::vector<T>> &GT)
    {
        FILE *F = nullptr;
        F = fopen(path_file.c_str(), "wb");
        if (F == NULL)
            throw std::runtime_error("error@write_gt_ivecs: File not found");

        size_t xxx;
        unsigned N = (unsigned)GT.size();
        for (size_t i = 0; i < (size_t)N; i++)
        {
            unsigned K = (unsigned)GT[i].size();
            xxx = fwrite(&K, sizeof(unsigned), 1, F);
            T *buffer = new T[K];
            for (size_t j = 0; j < (size_t)K; j++)
                buffer[j] = GT[i][j];
            xxx = fwrite(buffer, sizeof(T), K, F);
            delete[] buffer;
        }
        fclose(F);
    }

    // 读不同格式 base、query 数据集（float类型）
    inline float *read_data(const std::string &path_file, const std::string &format, unsigned &N, unsigned &Dim)
    {
        // 这里只考虑 float 类型的情况
        if (format == "bin")
            return read_data_bin<float>(path_file, N, Dim);
        else if (format == "vecs")
            return read_data_vecs<float>(path_file, N, Dim);
        else
            throw std::runtime_error("error@read_data: unsurported file format.");
    }

    // 写不同格式 base、query 数据集（float类型）
    inline void write_data(const std::string &path_file, const std::string &format, const float *data, unsigned N, unsigned Dim)
    {
        // 这里只考虑 float 类型的情况
        if (format == "bin")
            write_data_bin<const float>(path_file, data, N, Dim);
        else if (format == "vecs")
            write_data_vecs<const float>(path_file, data, N, Dim);
        else
            throw std::runtime_error("error@read_data: unsurported file format.");
    }

    // 读不同格式 gt 数据集（unsigned类型）
    inline std::vector<std::vector<unsigned>> read_gt(const std::string &path_file, const std::string &format, unsigned &N, unsigned &Dim)
    {
        // 这里只考虑 float 类型的情况
        if (format == "bin")
            return read_gt_ibin<unsigned>(path_file, N, Dim);
        else if (format == "vecs")
            return read_gt_ivecs<unsigned>(path_file, N, Dim);
        else
            throw std::runtime_error("error@read_gt: unsurported file format.");
    }

    template <typename T>
    inline void save_vector(const std::string &path_file, const std::vector<T> &P)
    {
        std::ofstream out(path_file, std::ios::out);
        if (!out.is_open())
            throw std::runtime_error("error:@save_vector: open file failed.");

        for (size_t i = 0; i < P.size(); ++i)
            out << P[i] << std::endl;
        out.close();
        std::cout << "save vector size: " << P.size() << std::endl;
    }

    template <typename T>
    inline void load_vector(const std::string &path_file, std::vector<T> &P)
    {
        std::ifstream in(path_file, std::ios::in);
        if (!in.is_open())
            throw std::runtime_error("error:@load_vector: open file failed.");

        T temp;
        while (in >> temp)
            P.push_back(temp);
        in.close();
        std::cout << "load vector size: " << P.size() << std::endl;
    }

    inline void *read_file(const std::string &path_file, std::size_t &size)
    {
        // 以二进制模式读取
        std::ifstream infile(path_file, std::ios::binary);
        if (!infile)
            throw std::runtime_error("error:@read_file: open file failed.");

        // 移动文件指针到末尾以获取文件大小
        infile.seekg(0, std::ios::end);
        size = infile.tellg();
        infile.seekg(0, std::ios::beg);

        // 读取文件
        char *data = new char[size];
        infile.read(data, size);
        if (!infile)
        {
            delete[] data;
            infile.close();
            throw std::runtime_error("error:@read_file: read file failed.");
        }
        infile.close();
        return data;
    }

    inline void read_file(const std::string &path_file, std::size_t &size, char *data, std::size_t max_size)
    {
        // 以二进制模式读取
        std::ifstream infile(path_file, std::ios::binary);
        if (!infile)
            throw std::runtime_error("error:@read_file: open file failed.");

        // 移动文件指针到末尾以获取文件大小
        infile.seekg(0, std::ios::end);
        size = infile.tellg();
        infile.seekg(0, std::ios::beg);

        // 读取文件
        // char *data = new char[size];
        if(size > max_size)
            throw std::runtime_error("error:@read_file: read file failed. size > max_size.");
        infile.read(data, size);
        if (!infile)
        {
            delete[] data;
            infile.close();
            throw std::runtime_error("error:@read_file: read file failed.");
        }
        infile.close();
        // return data;
    }

    inline void save_file(const std::string &path_file, const void *data, std::size_t size)
    {
        // 以二进制模式写入
        std::ofstream outfile(path_file, std::ios::binary);
        if (!outfile)
            throw std::runtime_error("error:@save_file: open file failed.");

        // 写入文件
        outfile.write(static_cast<const char *>(data), size);
        if (!outfile)
        {
            outfile.close();
            throw std::runtime_error("error:@save_file: save file failed.");
        }
        outfile.close();
    }

    inline void check_degree(std::vector<std::vector<unsigned>> &graph, unsigned &max_observed_degree)
    {
        uint64_t min_out_degree = UINT64_MAX, max_out_degree = 0, avg_out_degree = 0;
        for (size_t i = 0; i < graph.size(); i++)
        {
            min_out_degree = std::min(min_out_degree, graph[i].size());
            max_out_degree = std::max(max_out_degree, graph[i].size());
            avg_out_degree += graph[i].size();
        }
        avg_out_degree /= graph.size();
        std::cout << "min_out_degree: " << min_out_degree << std::endl;
        std::cout << "max_out_degree: " << max_out_degree << std::endl;
        std::cout << "avg_out_degree: " << avg_out_degree << std::endl;

        if (max_observed_degree != max_out_degree)
        {
            std::cout << "debug@check_degree: uncorrect max_observed_degree." << std::endl;
            // throw std::runtime_error("error@check_degree: uncorrect max_observed_degree.");
            max_observed_degree = max_out_degree;
        }
    }

    inline void load_nsg(const std::string &path_file, std::vector<std::vector<unsigned>> &graph, unsigned &max_observed_degree, unsigned &start)
    {
        std::ifstream in(path_file, std::ios::binary);
        if (!in.is_open() or in.fail())
        {
            in.close();
            throw std::runtime_error("error@load_nsg: graph_file open failed.");
        }
        in.read((char *)&max_observed_degree, sizeof(unsigned));
        in.read((char *)&start, sizeof(unsigned));
        while (!in.eof())
        {
            unsigned k;
            in.read((char *)&k, sizeof(unsigned));
            if (in.eof())
                break;
            std::vector<unsigned> tmp(k);
            in.read((char *)tmp.data(), k * sizeof(unsigned));
            // 1. Sort the neighbors
            std::sort(tmp.begin(), tmp.end());
            // 2. Use std::unique to remove adjacent duplicates
            auto last = std::unique(tmp.begin(), tmp.end());
            // 3. Erase the "removed" elements
            tmp.erase(last, tmp.end());
            graph.push_back(tmp);
        }
        in.close();
    }

    inline void load_vamana(const std::string &filename, size_t expected_num_points, std::vector<std::vector<uint32_t>> &_graph, uint32_t &_max_observed_degree, uint32_t &start)
    {
        size_t expected_file_size;
        size_t file_frozen_pts;
        size_t file_offset = 0; // will need this for single file format support

        std::ifstream in;
        in.exceptions(std::ios::badbit | std::ios::failbit);
        in.open(filename, std::ios::binary);
        in.seekg(file_offset, in.beg);
        in.read((char *)&expected_file_size, sizeof(size_t));
        in.read((char *)&_max_observed_degree, sizeof(uint32_t));
        in.read((char *)&start, sizeof(uint32_t));
        in.read((char *)&file_frozen_pts, sizeof(size_t));
        size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

        std::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree
                  << ", _start: " << start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

        std::cout << "Loading vamana graph " << filename << "..." << std::flush;

        _graph.resize(expected_num_points);

        size_t bytes_read = vamana_metadata_size;
        size_t cc = 0;
        uint32_t nodes_read = 0;
        while (bytes_read != expected_file_size)
        {
            uint32_t k;
            in.read((char *)&k, sizeof(uint32_t));
            if (k == 0)
            {
                std::cerr << "ERROR: Point found with no out-neighbours, point#" << nodes_read << std::endl;
            }

            cc += k;
            ++nodes_read;
            std::vector<uint32_t> tmp(k);
            tmp.reserve(k);
            in.read((char *)tmp.data(), k * sizeof(uint32_t));
            _graph[nodes_read - 1].swap(tmp);
            bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
            if (nodes_read % 10000000 == 0)
                std::cout << "." << std::flush;
        }
        std::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start << std::endl;
    }

} // namespace common
