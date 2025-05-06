#include "index.h"

namespace numaann
{
    Index::Index(const Parameters &para) : _para(para)
    {
        // load parameter
        std::string metric(_para.Get<std::string>("metric"));
        std::cout << "using metric " << metric << std::endl;

        // process......
        if (metric == "L2")
            _distance = new DistanceL2();
        else if (metric == "IP")
            _distance = new DistanceIP();
        else if (metric == "COS")
            _distance = new DistanceCOS();
        else
            throw std::runtime_error("error@Index: metric undefined.");
    }

    Index::~Index()
    {
        if (_distance != nullptr)
        {
            delete _distance;
            _distance = nullptr;
        }
        if (_gstore != nullptr)
        {
            delete _gstore;
            _gstore = nullptr;
        }

        for (size_t i = 0; i < _query_scratch.size(); i++)
        {
            delete _query_scratch[i];
            _query_scratch[i] = nullptr;
        }
        _query_scratch.clear();
        for (size_t i = 0; i < _query_scratch_distributed.size(); i++)
        {
            delete _query_scratch_distributed[i];
            _query_scratch_distributed[i] = nullptr;
        }
        _query_scratch_distributed.clear();
    }

    void Index::load_coarse_clusters()
    {
        // load parameter
        size_t bkmeans_K(_para.Get<std::size_t>("bkmeans_K"));
        std::string bkmeans_centroids_output_file(_para.Get<std::string>("bkmeans_centroids_output_file"));
        // std::string bkmeans_labels_output_file(_para.Get<std::string>("bkmeans_labels_output_file"));

        // process......
        if (_dimension == 0)
            throw std::runtime_error("error@load_coarse_clusters: _dimension UNDEFINED.");
        if (_coarse_clusters_data != nullptr)
            throw std::runtime_error("error@load_coarse_clusters: _coarse_clusters already loaded.");

        this->_coarse_clusters_num = bkmeans_K;
        this->_coarse_clusters_data.reset(new float[(size_t)_coarse_clusters_num * _dimension * sizeof(float)]);
        std::cout << "_coarse_clusters_num: " << _coarse_clusters_num << std::endl;

        FILE *F = nullptr;

        // read centroids
        F = fopen(bkmeans_centroids_output_file.c_str(), "r");
        if (F == NULL)
            throw std::runtime_error("error@load_coarse_clusters: _coarse_clusters file not found.");

        for (size_t k = 0; k < _coarse_clusters_num; ++k)
        {
            for (size_t d = 0; d < _dimension; d++)
            {
                int ret = fscanf(F, "%f", _coarse_clusters_data.get() + k * _dimension + d);
                if (ret != 1)
                    throw std::runtime_error("error@load_coarse_clusters: load_coarse_clusters failed.");
            }
        }
        fclose(F);

        // read labels
        // F = fopen(bkmeans_labels_output_file.c_str(), "r");
        // if (F == NULL)
        //     throw std::runtime_error("error@load_coarse_clusters: _coarse_clusters file not found.");

        // unsigned number;
        // while (fscanf(F, "%d", &number) != EOF)
        // {
        //     _coarse_clusters_label.push_back(number);
        // }
        // fclose(F);
    }

    unsigned Index::compute_closest_coarse_cluster(const float *point)
    {
        if (_distance == nullptr or _coarse_clusters_data == nullptr)
            throw std::runtime_error("error@compute_closest_coarse_cluster");
        return compute_closest_point(_coarse_clusters_data.get(), _coarse_clusters_num, _dimension, point, _distance);
    }

    // find the closest point in base_data
    unsigned Index::compute_closest_point(const float *base_data, size_t base_num, size_t base_dim, const float *query_point, const Distance *distance)
    {
        if (base_data == nullptr or base_num == 0 or base_dim == 0 or query_point == nullptr or distance == nullptr)
            throw std::runtime_error("error@compute_closest_point");

        unsigned closest_base_point = 0;
        double min_distance = distance->compare(base_data, query_point, (unsigned)base_dim);
        for (size_t i = 1; i < (size_t)base_num; i++)
        {
            double d = distance->compare(base_data + (uint64_t)i * base_dim, query_point, (unsigned)base_dim);
            if (d < min_distance)
            {
                closest_base_point = i;
                min_distance = d;
            }
        }
        return closest_base_point;
    }

    void Index::load_learn_data()
    {
        // load parameter
        std::string file_format(_para.Get<std::string>("file_format"));
        std::string learn_data_file(_para.Get<std::string>("learn_data_file"));

        // process......
        unsigned learn_num, learn_dim;
        float *learn_data = common::read_data(learn_data_file, file_format, learn_num, learn_dim);
        std::cout << "learn_num, learn_dim = " << learn_num << ", " << learn_dim << std::endl;

        this->_learn_num = learn_num;
        this->_learn_data.reset(learn_data);

        // check......
        if (!_learn_graph.empty() and _learn_graph.size() != _learn_num)
            throw std::runtime_error("error@load_learn_data: uncorrect learn_num.");
    }

    void Index::load_learn_graph()
    {
        // load parameter
        std::string path_file(_para.Get<std::string>("learn_graph_file"));

        // process......
        // common::load_nsg(path_file, this->_learn_graph, this->_learn_graph_R, this->_learn_graph_EP);
        common::load_vamana(path_file, this->_learn_num, this->_learn_graph, this->_learn_graph_R, this->_learn_graph_EP);
        common::check_degree(this->_learn_graph, this->_learn_graph_R);

        // check......
        if (_learn_num != _learn_graph.size())
            throw std::runtime_error("error@load_learn_graph: uncorrect graph size.");
    }

    void Index::generate_learn_projection()
    {
        // check......
        if (_base_in_bucket == nullptr)
            throw std::runtime_error("error@generate_learn_projection: _base_in_bucket is empty.");

        // load parameter
        std::string file_format(_para.Get<std::string>("file_format"));
        std::string learn_gt_file(_para.Get<std::string>("learn_gt_file"));

        // process......
        unsigned learn_gt_num, learn_gt_dim;
        std::vector<std::vector<unsigned>> learn_gt = common::read_gt(learn_gt_file, file_format, learn_gt_num, learn_gt_dim);
        std::cout << "learn_gt_num, learn_gt_dim = " << learn_gt_num << ", " << learn_gt_dim << std::endl;

        this->_learn_projection_in_base = learn_gt;
        this->_learn_in_bucket.reset(new unsigned[_learn_num]{0});
        this->_learn_local_ratio.reset(new float[_learn_num]{0});

        // 通过最大带权匹配划分
        // std::vector<std::vector<unsigned>> data_affinity;
        // data_affinity.resize(_learn_num);
        // for (vertex_id_t learn_id = 0; learn_id < _learn_num; learn_id++)
        // {
        //     // 使用 Top-N
        //     std::vector<unsigned> frequency(bucket_count, 0);
        //     unsigned TopN = std::min(1000, (int)learn_gt[learn_id].size());
        //     for (size_t i = 0; i < TopN; i++) // 可设置 gt 数量
        //     {
        //         vertex_id_t project_id = learn_gt[learn_id][i];
        //         unsigned bucket_id = _base_in_bucket[project_id];
        //         frequency[bucket_id]++;
        //     }
        //     data_affinity[learn_id] = frequency;
        // }
        // MaximumWeightMatch(data_affinity, _learn_in_bucket.get(), _learn_local_ratio.get());

        // 直接贪心划分
        for (vertex_id_t learn_id = 0; learn_id < _learn_num; learn_id++)
        {
            // 使用 Top-1
            // vertex_id_t project_id = gt[learn_id][0];
            // unsigned cluster_id = _base_in_bucket[project_id];
            // _learn_in_bucket[learn_id] = cluster_id;

            // 使用 Top-N
            std::vector<unsigned> frequency(bucket_count, 0);
            unsigned TopN = std::min(1000, (int)learn_gt[learn_id].size());
            for (size_t i = 0; i < TopN; i++) // 可设置 gt 数量
            {
                vertex_id_t project_id = learn_gt[learn_id][i];
                unsigned bucket_id = _base_in_bucket[project_id];
                frequency[bucket_id]++;
            }
            unsigned selected_bucket = 0;
            for (size_t bucket_id = 0; bucket_id < bucket_count; bucket_id++)
            {
                if (frequency[bucket_id] > frequency[selected_bucket])
                    selected_bucket = bucket_id;
            }
            _learn_in_bucket[learn_id] = selected_bucket;
            _learn_local_ratio[learn_id] = (float)frequency[selected_bucket] / TopN;

            // 使用 Top-N
            // std::vector<unsigned> tmp;
            // for (size_t i = 0; i < learn_gt[learn_id].size(); i++) // 可设置 gt 数量
            // {
            //     vertex_id_t project_id = learn_gt[learn_id][i];
            //     unsigned cluster_id = _base_in_bucket[project_id];
            //     tmp.push_back(cluster_id);
            // }
            // _learn_in_bucket[learn_id] = common::find_mode(tmp);
            // if(common::find_mode(tmp) != selected_cluster_id)
            //     std::cout << cluster_count[0] << ", " << cluster_count[1] << std::endl;
        }

        // 输出 _learn_in_bucket 统计信息
        std::vector<size_t> data_count(bucket_count, 0);
        for (vertex_id_t learn_id = 0; learn_id < _learn_num; learn_id++)
        {
            data_count[_learn_in_bucket[learn_id]]++;
        }
        std::cout << "learn data in buckets: ";
        for (size_t i = 0; i < bucket_count; i++)
            std::cout << data_count[i] << ", ";
        std::cout << std::endl;

        // check......
        if (_learn_data.get() == nullptr)
            throw std::runtime_error("error@generate_learn_projection: _learn_data is nullptr.");
        if (_learn_graph.empty())
            throw std::runtime_error("error@generate_learn_projection: _learn_graph is empty.");
    }

    unsigned Index::beam_search_learn_graph(const float *query, size_t L)
    {
        float *learn_data = _learn_data.get();

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{_learn_num, 0};

        unsigned tmp_l = 0;
        init_ids[tmp_l] = _learn_graph_EP;
        tmp_l++;
        flags[_learn_graph_EP] = true;

        L = tmp_l;
        for (unsigned i = 0; i < L; i++)
        {
            unsigned id = init_ids[i];
            float dist = _distance->compare(learn_data + _dimension * id, query, (unsigned)_dimension);
            retset[i] = Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;

            if (retset[k].flag)
            {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < _learn_graph[n].size(); ++m)
                {
                    unsigned id = _learn_graph[n][m];
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    float dist = _distance->compare(query, learn_data + _dimension * id, (unsigned)_dimension);
                    // if (dist >= retset[L - 1].distance)
                    if (L + 1 == retset.size() and dist >= retset[L - 1].distance) // 修改这行
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    if (L + 1 < retset.size())
                        ++L; // 加上这行

                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        return retset[0].id;
    }

    std::pair<unsigned, float> Index::search_affinity(const float *query)
    {
        unsigned should_run_on_node = compute_closest_coarse_cluster(query); // 对于 deep 数据集，直接通过计算距离query最近的cluster来调度性能更好，但这种方法对于 laion 数据集无效
        double dist = _distance->compare(_coarse_clusters_data.get() + (uint64_t)should_run_on_node * _dimension, query, (unsigned)_dimension);
        return std::make_pair(should_run_on_node, -dist);
    }

    std::pair<unsigned, float> Index::search_affinity(unsigned learn_index_res)
    {
        unsigned should_run_on_node = _learn_in_bucket[learn_index_res];
        float data_affinity = _learn_local_ratio[learn_index_res];
        return std::make_pair(should_run_on_node, data_affinity);
    }

    std::pair<unsigned, float> Index::search_affinity(const std::vector<unsigned> &gt)
    {
        // 使用 Top-N
        std::vector<unsigned> frequency(bucket_count, 0);
        unsigned TopN = std::min(1000, (int)gt.size());
        for (size_t i = 0; i < TopN; i++) // 可设置 gt 数量
        {
            vertex_id_t project_id = gt[i];
            unsigned bucket_id = _base_in_bucket[project_id];
            frequency[bucket_id]++;
        }
        unsigned selected_bucket = 0;
        for (size_t bucket_id = 0; bucket_id < bucket_count; bucket_id++)
        {
            if (frequency[bucket_id] > frequency[selected_bucket])
                selected_bucket = bucket_id;
        }
        unsigned should_run_on_node = selected_bucket;
        float data_affinity = (float)frequency[selected_bucket] / TopN;
        return std::make_pair(should_run_on_node, data_affinity);
    }

    void Index::add_adaptive_ep(unsigned learn_index_res, unsigned adaptive_ep_num, std::vector<item_t> &init_ids)
    {
        if (learn_index_res >= _learn_projection_in_base.size())
            throw std::runtime_error("error@add_adaptive_ep: learn_index_res is out of range.");
        if (adaptive_ep_num > _learn_projection_in_base[learn_index_res].size())
            throw std::runtime_error("error@add_adaptive_ep: adaptive_ep_num is out of range.");
        for (unsigned i = 0; i < adaptive_ep_num; i++)
        {
            unsigned base_id = _learn_projection_in_base[learn_index_res][i];
            item_t adaptive_ep = std::make_pair((server_id_t)_base_in_bucket[base_id], (local_id_t)_base_to_lid[base_id]);
            init_ids.push_back(adaptive_ep);
        }
    }

    void Index::load_base_data()
    {
        // load parameter
        std::string file_format(_para.Get<std::string>("file_format"));
        std::string base_file(_para.Get<std::string>("base_file"));

        // process......
        unsigned base_num, base_dim;
        float *base_data = common::read_data(base_file, file_format, base_num, base_dim);
        std::cout << "base_num, base_dim = " << base_num << ", " << base_dim << std::endl;

        this->_base_num = base_num;
        this->_dimension = base_dim;
        this->_base_data.reset(base_data);

        // check......
        if (!_base_graph.empty() and _base_graph.size() != _base_num)
            throw std::runtime_error("error@load_base_data: uncorrect base_num.");
    }

    void Index::load_base_graph()
    {
        // load parameter
        std::string path_file(_para.Get<std::string>("graph_file"));

        // process......
        // common::load_nsg(path_file, this->_base_graph, this->_base_graph_R, this->_base_graph_EP);
        common::load_vamana(path_file, this->_base_num, this->_base_graph, this->_base_graph_R, this->_base_graph_EP);
        common::check_degree(this->_base_graph, this->_base_graph_R);

        // check......
        if (_base_num != _base_graph.size())
            throw std::runtime_error("error@load_graph: uncorrect graph size.");
    }

    void Index::generate_base_index()
    {
        if (_base_data.get() == nullptr)
            throw std::runtime_error("error@generate_base_index: _base_data is nullptr.");

        if (_base_graph.empty())
            throw std::runtime_error("error@generate_base_index: _base_graph is empty.");

        size_t data_bytes_len = _dimension * sizeof(float);
        size_t label_bytes_len = sizeof(unsigned);
        size_t neighbor_bytes_len = (1 + _base_graph_R) * sizeof(unsigned);

        this->_data_offset = 0;
        this->_label_offset = data_bytes_len;
        this->_neighbor_offset = data_bytes_len + label_bytes_len;
        this->_element_size = data_bytes_len + label_bytes_len + neighbor_bytes_len; // base data + label + neighbor_count + neighbor_list
        this->_element_num = _base_num;

        std::cout << "data_bytes_len: " << data_bytes_len << std::endl;
        std::cout << "label_bytes_len: " << label_bytes_len << std::endl;
        std::cout << "neighbor_bytes_len: " << neighbor_bytes_len << std::endl;
        std::cout << "_element_size: " << _element_size << std::endl;
        std::cout << "_element_num: " << _element_num << std::endl;

        this->_base_index.reset((char *)malloc((uint64_t)_element_num * _element_size));
        if (_base_index.get() == nullptr)
            throw std::runtime_error("error@generate_base_index: malloc failed.");

        for (vertex_id_t cur_id = 0; cur_id < _element_num; cur_id++)
        {
            char *cur_node_offset = this->_base_index.get() + (uint64_t)cur_id * _element_size;

            std::memcpy(cur_node_offset + _data_offset, _base_data.get() + (uint64_t)cur_id * _dimension, _dimension * sizeof(float));
            std::memcpy(cur_node_offset + _label_offset, &cur_id, sizeof(unsigned));

            unsigned *neighbor_lsit = (unsigned *)(cur_node_offset + _neighbor_offset);
            *neighbor_lsit = _base_graph[cur_id].size();
            neighbor_lsit++;
            for (size_t n = 0; n < _base_graph[cur_id].size(); n++)
            {
                *neighbor_lsit = _base_graph[cur_id][n];
                neighbor_lsit++;
            }
        }
        this->_base_index_R = _base_graph_R;
        this->_base_index_EP = _base_graph_EP;
    }

    void Index::generate_base_index_on_buckets()
    {
        if (_base_data.get() == nullptr)
            throw std::runtime_error("error@generate_base_index: _base_data is nullptr.");

        if (_base_graph.empty())
            throw std::runtime_error("error@generate_base_index: _base_graph is empty.");

        this->bucket_count = _para.Get<std::size_t>("bucket_count");

        size_t data_bytes_len = _dimension * sizeof(float);
        size_t label_bytes_len = sizeof(unsigned);
        // size_t neighbor_bytes_len = sizeof(unsigned) + _base_graph_R * (sizeof(server_id_t) + sizeof(local_id_t));
        size_t neighbor_bytes_len = sizeof(unsigned) + _base_graph_R * (sizeof(vertex_id_t) + sizeof(server_id_t) + sizeof(local_id_t));

        this->_data_offset = 0;
        this->_label_offset = data_bytes_len;
        this->_neighbor_offset = data_bytes_len + label_bytes_len;
        this->_element_size = data_bytes_len + label_bytes_len + neighbor_bytes_len; // base data + label + neighbor_count + neighbor_list(node_id, local_id)
        this->_element_num = _base_num;

        std::cout << "data_bytes_len: " << data_bytes_len << std::endl;
        std::cout << "label_bytes_len: " << label_bytes_len << std::endl;
        std::cout << "neighbor_bytes_len: " << neighbor_bytes_len << std::endl;
        std::cout << "_element_size: " << _element_size << std::endl;
        std::cout << "_element_num: " << _element_num << std::endl;

        this->data_num.resize(bucket_count, 0);
        std::vector<item_t> place(_element_num);

        // 数据划分
        this->_base_in_bucket.reset(new unsigned[_base_num]{0});
        this->_base_to_lid.resize(_base_num, 0);
        for (vertex_id_t id = 0; id < _base_num; id++)
        {
            // server_id_t bucket_id = 0;
            // server_id_t bucket_id = id % bucket_count; // random partition
            server_id_t bucket_id = compute_closest_coarse_cluster(_base_data.get() + (uint64_t)id * _dimension);
            _base_in_bucket[id] = bucket_id;
        }
        for (vertex_id_t id = 0; id < _base_num; id++)
        {
            server_id_t bucket_id = _base_in_bucket[id];
            local_id_t local_id = data_num[bucket_id];
            place[id] = std::make_pair(bucket_id, local_id);
            data_num[bucket_id]++;
            _base_to_lid[id] = local_id;
        }
        for (size_t bucket_id = 0; bucket_id < bucket_count; bucket_id++)
        {
            std::cout << "bucket#" << (unsigned)bucket_id << ", alloc points: " << data_num[bucket_id] << std::endl;
        }

        // 分配空间
        this->_memeory_buckets = (char **)malloc(bucket_count * sizeof(char *));
        for (size_t bucket_id = 0; bucket_id < bucket_count; bucket_id++)
        {
            size_t bucket_size = (size_t)_element_size * data_num[bucket_id];
            _memeory_buckets[bucket_id] = nullptr;
            _memeory_buckets[bucket_id] = (char *)malloc(bucket_size);
            if (_memeory_buckets[bucket_id] == nullptr)
                throw std::runtime_error("error@generate_base_index_with_bucket: alloc failed.");
            // printf("%p ", _memeory_buckets[bucket_id]);
            std::cout << "bucket#" << (unsigned)bucket_id << " alloc bytes: " << bucket_size << std::endl;
        }

        // 重排数据(多线程)
        // #pragma omp parallel for schedule(dynamic, 1)
        for (vertex_id_t cur_id = 0; cur_id < _element_num; cur_id++)
        {
            server_id_t bucket_id = place[cur_id].first;
            local_id_t local_id = place[cur_id].second;
            char *destination = _memeory_buckets[bucket_id] + (uint64_t)local_id * _element_size;

            std::memcpy(destination + _data_offset, _base_data.get() + (uint64_t)cur_id * _dimension, _dimension * sizeof(float));
            std::memcpy(destination + _label_offset, &cur_id, sizeof(unsigned));

            unsigned *neighbor_lsit = (unsigned *)(destination + _neighbor_offset);
            *neighbor_lsit = _base_graph[cur_id].size();
            neighbor_lsit++;

            destination = (char *)neighbor_lsit;
            std::vector<item_t> neighbors;
            for (vertex_id_t neighbor_id : _base_graph[cur_id])
            {
                std::memcpy(destination, &neighbor_id, sizeof(vertex_id_t));
                destination += sizeof(vertex_id_t);
                server_id_t bucket_id = place[neighbor_id].first;
                local_id_t local_id = place[neighbor_id].second;
                //     neighbors.push_back(make_pair(bucket_id, local_id));
                // }
                // std::sort(neighbors.begin(), neighbors.end());
                // for (item_t neighbor_item : neighbors)
                // {
                //     server_id_t bucket_id = neighbor_item.first;
                //     local_id_t local_id = neighbor_item.second;
                std::memcpy(destination, &bucket_id, sizeof(server_id_t));
                destination += sizeof(server_id_t);
                std::memcpy(destination, &local_id, sizeof(local_id_t));
                destination += sizeof(local_id_t);
            }
        }
        this->_base_index_R = _base_graph_R;
        this->_membkt_EP = place[_base_graph_EP];
    }

    void Index::save_base_index_on_buckets()
    {
        // load parameter
        std::string filename_prefix(_para.Get<std::string>("filename_prefix"));

        // process......
        std::string meta_filepath = filename_prefix + ".meta";
        std::ofstream outfile(meta_filepath);
        if (!outfile)
            throw std::runtime_error("error:@meta_filepath: open file failed.");

        outfile << this->_base_num << std::endl;
        outfile << this->_dimension << std::endl;

        outfile << this->_element_num << std::endl;
        outfile << this->_element_size << std::endl;
        outfile << this->_data_offset << std::endl;
        outfile << this->_label_offset << std::endl;
        outfile << this->_neighbor_offset << std::endl;

        outfile << this->_base_index_R << std::endl;

        outfile << (unsigned)this->_membkt_EP.first << std::endl;
        outfile << (unsigned)this->_membkt_EP.second << std::endl;
        outfile.close();

        for (size_t bucket_id = 0; bucket_id < this->bucket_count; bucket_id++)
        {
            std::string bucket_filepath = filename_prefix + ".bucket_" + std::to_string(bucket_id);
            common::save_file(bucket_filepath, this->_memeory_buckets[bucket_id], this->data_num[bucket_id] * this->_element_size);
        }

        std::string partition_filepath = filename_prefix + ".partition";
        common::save_file(partition_filepath, this->_base_in_bucket.get(), this->_base_num * sizeof(unsigned));

        std::string lid_filepath = filename_prefix + ".lid";
        common::save_vector(lid_filepath, _base_to_lid);

        std::string data_num_filepath = filename_prefix + ".data_num";
        common::save_vector(data_num_filepath, data_num);
    }

    void Index::load_base_index_distributed(int sid, Mem *mem)
    {
        this->bucket_count = _para.Get<std::size_t>("bucket_count");
        ASSERT(bucket_count == Global::num_servers);

        // load parameter
        std::string filename_prefix(_para.Get<std::string>("filename_prefix"));

        // process......
        std::string meta_filepath = filename_prefix + ".meta";
        std::ifstream infile(meta_filepath);
        if (!infile)
            throw std::runtime_error("error:@meta_filepath: open file failed.");

        infile >> this->_base_num;
        infile >> this->_dimension;

        infile >> this->_element_num;
        infile >> this->_element_size;
        infile >> this->_data_offset;
        infile >> this->_label_offset;
        infile >> this->_neighbor_offset;

        infile >> this->_base_index_R;

        unsigned tmp;
        infile >> tmp;
        this->_membkt_EP.first = (server_id_t)tmp;
        infile >> tmp;
        this->_membkt_EP.second = (local_id_t)tmp;
        infile.close();

        char *data = mem->kvstore();
        size_t data_size, max_size = mem->kvstore_size();

        std::string bucket_filepath = filename_prefix + ".bucket_" + std::to_string(sid);
        common::read_file(bucket_filepath, data_size, data, max_size);

        // 判断分配的内存是否足够存放图数据
        // ASSERT(data != nullptr);
        // ASSERT(data_size < mem->kvstore_size());
        // std::memcpy(mem->kvstore(), data, data_size);
        // delete[] data;
        data = nullptr;

        logstream(LOG_EMPH) << "#" << sid << ": load data_size " << data_size << LOG_endl;
        logstream(LOG_EMPH) << "#" << sid << ": load data_num " << data_size / _element_size << LOG_endl;

        logstream(LOG_EMPH) << "#" << sid << ": load _dimension: " << _dimension << LOG_endl;
        logstream(LOG_EMPH) << "#" << sid << ": load _element_num: " << _element_num << std::endl;
        logstream(LOG_EMPH) << "#" << sid << ": load _element_size: " << _element_size << std::endl;
        logstream(LOG_EMPH) << "#" << sid << ": load _base_index_R: " << _base_index_R << std::endl;
        logstream(LOG_EMPH) << "#" << sid << ": load _membkt_EP: (" << (unsigned)_membkt_EP.first << ", " << _membkt_EP.second << ")" << LOG_endl;

        _gstore = new GStore(sid, mem, _element_size, _base_index_R);

        std::string partition_filepath = filename_prefix + ".partition";
        data = (char *)common::read_file(partition_filepath, data_size);

        ASSERT(data != nullptr);
        this->_base_in_bucket.reset(new unsigned[_base_num]{0});
        std::memcpy(_base_in_bucket.get(), data, data_size);
        delete[] data;
        data = nullptr;

        std::string lid_filepath = filename_prefix + ".lid";
        common::load_vector(lid_filepath, _base_to_lid);

        std::string data_num_filepath = filename_prefix + ".data_num";
        common::load_vector(data_num_filepath, data_num);

        /* 原始排列 */
        // if (sid == 0)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_0", data_size);
        // else if (sid == 1)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1", data_size);
        /* 使用gorder重排 */
        // if (sid == 0)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_0_gorder", data_size);
        // else if (sid == 1)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1_gorder", data_size);
        /* 使用remote_neighbor_order重排 */
        // if (sid == 0)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_0_remote_neighbor_order", data_size);
        // else if (sid == 1)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1_remote_neighbor_order", data_size);
        /* 使用remote_neighbor_order_only_remote重排 */
        // if (sid == 0)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_0_remote_neighbor_order_only_remote", data_size);
        // else if (sid == 1)
        //     data = (char *)common::read_file("/home/cy/experiment/RDMA-ANNS/tmp/inpute_server_1_remote_neighbor_order_only_remote", data_size);

        // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
        // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
        // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2960); // 临时赋值(remote_neighbor_order)
        // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3685910);    // 临时赋值(remote_neighbor_order_only_remote)

        // 验证
        // if (this->_membkt_EP != ep)
        //     throw std::runtime_error("error@check failed.");
        // for (size_t i = 0; i < data_size / _element_size; i++)
        // {
        //     if (*(unsigned *)(mem->kvstore() + i * _element_size + _label_offset) != *(unsigned *)(data + i * _element_size + _label_offset))
        //         throw std::runtime_error("error@check failed.");
        // }
        // logstream(LOG_DEBUG) << "#" << sid << ": check success. " << LOG_endl;
        // delete[] data;
        // data = nullptr;
    }

    void Index::set_cache(float *query_data, unsigned query_num, const std::vector<unsigned> &query_bucket, uint32_t numThreads, size_t num_nodes_to_cache)
    {
        std::vector<std::vector<uint32_t>> access_count;
        access_count.resize(Global::num_servers);
        for (size_t ssid = 0; ssid < Global::num_servers; ssid++)
        {
            access_count[ssid].resize(data_num[ssid], 0);
        }

        // profile
        std::cout << "profile......" << std::endl;
        common::Timer timer;
#pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
        for (uint64_t tmp = 0; tmp < query_bucket.size(); tmp++)
        {
            uint64_t i = query_bucket[tmp];
            int threadId = omp_get_thread_num();
            search_base_index_distributed(threadId, query_data + i * _dimension, access_count);
        }
        float seconds = timer.elapsed_seconds();
        std::cout << "profile finished. time(s): " << seconds << std::endl;

        // set_cache
        _gstore->set_cache(access_count, data_num, num_nodes_to_cache);
    }

    /* 单机搜索函数 */
    void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices, float *distances, common::QueryStats *stats)
    {
        diskann::InMemQueryScratch *scratch = _query_scratch[tid];
        scratch->clear();

        diskann::NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        std::vector<uint32_t> &id_scratch = scratch->id_scratch();
        std::vector<char *> &vertex_scratch = scratch->vertex_scratch();
        tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();

        auto is_not_visited = [this, &inserted_into_pool_rs](const uint32_t id)
        {
            return inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
        };

        auto insert_into_visited = [this, &inserted_into_pool_rs](const uint32_t id)
        {
            inserted_into_pool_rs.insert(id);
        };

        auto get_vertex = [this](const uint32_t id)
        {
            return (char *)_base_index.get() + (uint64_t)id * _element_size;
        };

        auto update_stats = [this, stats](const std::vector<uint32_t> &id_scratch)
        {
            stats->n_hops++;
            stats->n_cmps += id_scratch.size();
        };

        std::vector<unsigned> init_ids;
        init_ids.push_back(_base_index_EP);

        // traverse graph
        id_scratch.clear();
        vertex_scratch.clear();
        for (unsigned id : init_ids)
        {
            if (is_not_visited(id))
            {
                insert_into_visited(id);
                id_scratch.push_back(id);
            }
        }
        if (stats)
            update_stats(id_scratch);
        // Read vector
        for (unsigned id : id_scratch)
        {
            vertex_scratch.push_back(get_vertex(id));
        }
        // compute distance
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            unsigned id = id_scratch[m];
            float *base = (float *)(vertex_scratch[m] + _data_offset);
            float dist = _distance->compare(base, query, (unsigned)_dimension);
            bool succ = best_L_nodes.insert(diskann::Neighbor(id, dist));
        }

        while (best_L_nodes.has_unexpanded_node())
        {
            unsigned n = best_L_nodes.closest_unexpanded().id;

            // traverse graph
            id_scratch.clear();
            vertex_scratch.clear();
            unsigned *neighbors = (unsigned *)(get_vertex(n) + _neighbor_offset);
            unsigned MaxM = *neighbors;
            neighbors++;
            for (unsigned m = 0; m < MaxM; ++m)
            {
                unsigned id = neighbors[m];
                if (is_not_visited(id))
                {
                    insert_into_visited(id);
                    id_scratch.push_back(id);
                }
            }
            if (stats)
                update_stats(id_scratch);
            // Read vector
            for (unsigned id : id_scratch)
            {
                vertex_scratch.push_back(get_vertex(id));
            }
            // compute distance
            for (size_t m = 0; m < id_scratch.size(); ++m)
            {
                unsigned id = id_scratch[m];
                float *base = (float *)(vertex_scratch[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
                bool succ = best_L_nodes.insert(diskann::Neighbor(id, dist));
            }
        }
        for (size_t i = 0; i < K; i++)
        {
            indices[i] = *(unsigned *)(get_vertex(best_L_nodes[i].id) + _label_offset);
            distances[i] = best_L_nodes[i].distance;
        }
    }

    /* 单机搜索函数 */
    /* 通过 _memeory_buckets 访存和使用 item 会损失一点性能 */
    void Index::search_base_index_on_buckets(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    {
        dsmann::InMemQueryScratch *scratch = _query_scratch_distributed[tid];
        scratch->clear();

        dsmann::NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        std::vector<item_t> &item_scratch = scratch->item_scratch();
        std::vector<char *> &vertex_scratch = scratch->vertex_scratch();
        std::vector<tsl::robin_set<local_id_t>> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();

        auto is_not_visited = [this, &inserted_into_pool_rs](const item_t &item)
        {
            return inserted_into_pool_rs[item.first].find(item.second) == inserted_into_pool_rs[item.first].end();
        };

        auto insert_into_visited = [this, &inserted_into_pool_rs](const item_t &item)
        {
            inserted_into_pool_rs[item.first].insert(item.second);
        };

        auto get_vertex = [this](const item_t &item)
        {
            return (char *)_memeory_buckets[item.first] + (uint64_t)item.second * _element_size;
        };

        std::vector<item_t> init_ids;
        init_ids.push_back(_membkt_EP);

        // traverse graph
        item_scratch.clear();
        vertex_scratch.clear();
        for (const item_t &item : init_ids)
        {
            if (is_not_visited(item))
            {
                insert_into_visited(item);
                item_scratch.push_back(item);
            }
        }
        // Read vector
        for (const item_t &item : item_scratch)
        {
            vertex_scratch.push_back(get_vertex(item));
        }
        // compute distance
        for (size_t m = 0; m < item_scratch.size(); ++m)
        {
            const item_t &item = item_scratch[m];
            float *base = (float *)(vertex_scratch[m] + _data_offset);
            float dist = _distance->compare(base, query, (unsigned)_dimension);
            bool succ = best_L_nodes.insert(dsmann::Neighbor(item, dist));
        }

        while (best_L_nodes.has_unexpanded_node())
        {
            item_t n_item = best_L_nodes.closest_unexpanded().item;

            // traverse graph
            item_scratch.clear();
            vertex_scratch.clear();
            char *tmp = get_vertex(n_item) + _neighbor_offset;
            unsigned MaxM = *((unsigned *)tmp);
            tmp += sizeof(unsigned);
            for (unsigned m = 0; m < MaxM; ++m)
            {
                vertex_id_t vid = *((vertex_id_t *)tmp);
                tmp += sizeof(vertex_id_t);

                item_t m_item;
                m_item.first = *((server_id_t *)tmp);
                tmp += sizeof(server_id_t);
                m_item.second = *((local_id_t *)tmp);
                tmp += sizeof(local_id_t);
                if (is_not_visited(m_item))
                {
                    insert_into_visited(m_item);
                    item_scratch.push_back(m_item);
                }
            }
            // Read vector
            for (const item_t &item : item_scratch)
            {
                vertex_scratch.push_back(get_vertex(item));
            }
            // compute distance
            for (size_t m = 0; m < item_scratch.size(); ++m)
            {
                const item_t &item = item_scratch[m];
                float *base = (float *)(vertex_scratch[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
                bool succ = best_L_nodes.insert(dsmann::Neighbor(item, dist));
            }
        }
        for (size_t i = 0; i < K; i++)
        {
            indices[i] = *(unsigned *)(get_vertex(best_L_nodes[i].item) + _label_offset);
        }
    }

    void Index::search_base_index_distributed(int tid, const float *query, std::vector<std::vector<uint32_t>> &access_count)
    {
        auto update_access_count = [this, &access_count](const item_t &item)
        {
            reinterpret_cast<std::atomic<uint32_t> &>(access_count[item.first][item.second]).fetch_add(1);
        };

        dsmann::InMemQueryScratch *scratch = _query_scratch_distributed[tid];
        scratch->clear();

        dsmann::NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        std::vector<item_t> &item_scratch = scratch->item_scratch();
        std::vector<char *> &vertex_scratch = scratch->vertex_scratch();
        std::vector<tsl::robin_set<local_id_t>> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();

        auto is_not_visited = [this, &inserted_into_pool_rs](const item_t &item)
        {
            return inserted_into_pool_rs[item.first].find(item.second) == inserted_into_pool_rs[item.first].end();
        };

        auto insert_into_visited = [this, &inserted_into_pool_rs](const item_t &item)
        {
            inserted_into_pool_rs[item.first].insert(item.second);
        };

        _gstore->reset_thd_ctx(tid);

        item_t ep = this->_membkt_EP;
        std::vector<item_t> init_ids;
        init_ids.push_back(ep);

        // traverse graph
        for (const item_t &item : init_ids)
        {
            if (is_not_visited(item))
            {
                insert_into_visited(item);
                item_scratch.push_back(item);
            }
        }
        // Read vector
        _gstore->get_vertex_batch(tid, item_scratch.data(), item_scratch.size(), vertex_scratch);
        // compute distance
        for (size_t i = 0; i < item_scratch.size(); ++i)
        {
            item_t item = item_scratch[i];
            update_access_count(item);
            float *base = (float *)(vertex_scratch[i] + _data_offset);
            float dist = _distance->compare(base, query, (unsigned)_dimension);
            bool succ = best_L_nodes.insert(dsmann::Neighbor(item, dist));
            if (succ)
                _gstore->update_thd_cache(tid, item, vertex_scratch[i]);
        }
        item_scratch.clear();
        vertex_scratch.clear();

        while (best_L_nodes.has_unexpanded_node())
        {
            item_scratch.clear();
            vertex_scratch.clear();

            item_t n_item = best_L_nodes.closest_unexpanded().item;

            // traverse graph
            char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
            unsigned MaxM = *((unsigned *)tmp);
            tmp += sizeof(unsigned);
            for (unsigned m = 0; m < MaxM; ++m)
            {
                item_t m_item;
                m_item.first = *((server_id_t *)tmp);
                tmp += sizeof(server_id_t);
                m_item.second = *((local_id_t *)tmp);
                tmp += sizeof(local_id_t);
                if (is_not_visited(m_item))
                {
                    insert_into_visited(m_item);
                    item_scratch.push_back(m_item);
                }
            }
            // Read vector
            _gstore->get_vertex_batch(tid, item_scratch.data(), item_scratch.size(), vertex_scratch, true); // for Asyn I/O
            _gstore->get_vertex_batch_wait(tid);                                                            // wait for Asyn I/O

            // Compute distance
            for (unsigned m = 0; m < item_scratch.size(); ++m)
            {
                item_t m_item = item_scratch[m];
                update_access_count(m_item);
                float *base = (float *)(vertex_scratch[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
                bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
                if (succ)
                    _gstore->update_thd_cache(tid, m_item, vertex_scratch[m]); // for cache
            }
        }
    }

    void Index::test_compute(int tid, const float *query, uint64_t NUM_ITERATIONS, uint64_t batch_size)
    {
        dsmann::InMemQueryScratch *scratch = _query_scratch_distributed[tid];
        scratch->clear();

        std::vector<item_t> &item_scratch = scratch->item_scratch();
        std::vector<char *> &vertex_scratch = scratch->vertex_scratch();

        _gstore->reset_thd_ctx(tid);

        // 每个线程有自己的随机数生成器和分布， 注意rand()是线程不安全的
        thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<> dis(0, 20 * 1000 * 1000);

        for (uint64_t i = 0; i < NUM_ITERATIONS; i++)
        {
            if (NUM_ITERATIONS % 10000 == 0)
                _gstore->reset_thd_ctx(tid);

            item_scratch.clear();
            vertex_scratch.clear();
            for (unsigned m = 0; m < batch_size; ++m)
            {
                item_t m_item;
                if (m % 20 == 0) // 控制远程访问比例
                    m_item.first = 1; // remote
                else
                    m_item.first = 0;   // local
                // m_item.second = rand() % 20 * 1000 * 1000; // 线程不安全
                m_item.second = dis(generator); // 线程安全
                item_scratch.push_back(m_item);
            }
            _gstore->get_vertex_batch(tid, item_scratch.data(), item_scratch.size(), vertex_scratch, true); // for Asyn I/O
            _gstore->get_vertex_batch_wait(tid);                                                            // wait for Asyn I/O
            for (unsigned m = 0; m < item_scratch.size(); ++m)
            {
                item_t m_item = item_scratch[m];
                float *base = (float *)(vertex_scratch[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
            }
        }
    }

    /* 新的实现，注意对于同一个query，依赖松弛后，在不同的server上执行的搜索路径可能不同，因此召回值也可能不同 */
    void Index::search_base_index_distributed(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices, unsigned learn_index_res, unsigned adaptive_ep_num, int relax, common::QueryStats *stats)
    {
        // auto start = std::chrono::high_resolution_clock::now();
        dsmann::InMemQueryScratch *scratch = _query_scratch_distributed[tid];
        scratch->clear();

        dsmann::NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        std::vector<item_t> &item_scratch = scratch->item_scratch();
        std::vector<char *> &vertex_scratch = scratch->vertex_scratch();
        std::vector<tsl::robin_set<local_id_t>> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();

        auto is_not_visited = [this, &inserted_into_pool_rs](const item_t &item)
        {
            return inserted_into_pool_rs[item.first].find(item.second) == inserted_into_pool_rs[item.first].end();
        };

        auto insert_into_visited = [this, &inserted_into_pool_rs](const item_t &item)
        {
            inserted_into_pool_rs[item.first].insert(item.second);
        };

        auto update_stats = [this, stats](const std::vector<item_t> &item_scratch)
        {
            stats->n_hops++;
            stats->n_cmps += item_scratch.size();
            for (auto item : item_scratch)
            {
                if (item.first == _gstore->sid)
                    stats->n_cmps_local++;
                else
                    stats->n_cmps_remote++;
            }
        };

        _gstore->reset_thd_ctx(tid);

        boost::circular_buffer<std::vector<item_t>> queue_item_batch(relax + 1);
        boost::circular_buffer<std::vector<char *>> queue_vertex_batch(relax + 1);

        item_t ep = this->_membkt_EP;
        std::vector<item_t> init_ids;
        init_ids.push_back(ep);

        if (adaptive_ep_num > 0)
            add_adaptive_ep(learn_index_res, adaptive_ep_num, init_ids);

        // 依赖松弛需要至少 relax + 1 个ep
        ASSERT(init_ids.size() >= relax + 1);
        // for (size_t i = 0; i < relax; i++)
        // {
        //     ep.second++;
        //     init_ids.push_back(ep);
        // }

        // traverse graph
        for (const item_t &item : init_ids)
        {
            if (is_not_visited(item))
            {
                insert_into_visited(item);
                item_scratch.push_back(item);
            }
        }
        if (stats)
            update_stats(item_scratch);
        // Read vector
        _gstore->get_vertex_batch(tid, item_scratch.data(), item_scratch.size(), vertex_scratch);
        // compute distance
        for (size_t i = 0; i < item_scratch.size(); ++i)
        {
            item_t item = item_scratch[i];
            float *base = (float *)(vertex_scratch[i] + _data_offset);
            float dist = _distance->compare(base, query, (unsigned)_dimension);
            bool succ = best_L_nodes.insert(dsmann::Neighbor(item, dist));
            if (succ)
                _gstore->update_thd_cache(tid, item, vertex_scratch[i]);
        }
        item_scratch.clear();
        vertex_scratch.clear();
        pq.inti_dist_vec(tid, query);

        while (best_L_nodes.has_unexpanded_node())
        {
            item_scratch.clear();
            vertex_scratch.clear();

            item_t n_item = best_L_nodes.closest_unexpanded().item;

            // traverse graph
            char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
            unsigned MaxM = *((unsigned *)tmp);
            tmp += sizeof(unsigned);
            for (unsigned m = 0; m < MaxM; ++m)
            {
                vertex_id_t vid = *((vertex_id_t *)tmp);
                tmp += sizeof(vertex_id_t);
                item_t m_item;
                m_item.first = *((server_id_t *)tmp);
                tmp += sizeof(server_id_t);
                m_item.second = *((local_id_t *)tmp);
                tmp += sizeof(local_id_t);
                if (is_not_visited(m_item))
                {
                    insert_into_visited(m_item);
                    // item_scratch.push_back(m_item);
                    if (m_item.first != _gstore->sid)
                    {
                        bool flag = (best_L_nodes.size() < best_L_nodes.capacity()) or (pq.compute_dist(tid, vid) < best_L_nodes[best_L_nodes.size()-1].distance*1.1);
                        if(flag){
                            item_scratch.push_back(m_item);
                        }
                    }
                    else
                    {
                        item_scratch.push_back(m_item);
                    }
                }
            }
            if (stats)
                update_stats(item_scratch);
            // Read vector
            _gstore->get_vertex_batch(tid, item_scratch.data(), item_scratch.size(), vertex_scratch, true); // for Asyn I/O
            queue_item_batch.push_back(item_scratch);
            queue_vertex_batch.push_back(vertex_scratch);

            // Compute distance(local)
            for (unsigned m = 0; m < item_scratch.size(); ++m)
            {
                item_t m_item = item_scratch[m];
                if (m_item.first != _gstore->sid)
                    continue;
                float *base = (float *)(vertex_scratch[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
                bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
                if (succ)
                    _gstore->update_thd_cache(tid, m_item, vertex_scratch[m]); // for cache
            }

            if (relax > 0)
            {
                relax--;
                continue;
            }
            std::vector<item_t> &item_scratch_ref = queue_item_batch.front();
            std::vector<char *> &vertex_scratch_ref = queue_vertex_batch.front();
            _gstore->get_vertex_batch_wait(tid); // wait for Asyn I/O

            // Compute distance(remote)
            for (unsigned m = 0; m < item_scratch_ref.size(); ++m)
            {
                item_t m_item = item_scratch_ref[m];
                if (m_item.first == _gstore->sid)
                    continue;
                float *base = (float *)(vertex_scratch_ref[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
                bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
                if (succ)
                    _gstore->update_thd_cache(tid, m_item, vertex_scratch_ref[m]); // for cache
            }
            queue_item_batch.pop_front();
            queue_vertex_batch.pop_front();
        }

        /* 处理队列中剩余的vertex */
        while (!queue_item_batch.empty())
        {
            std::vector<item_t> &item_scratch_ref = queue_item_batch.front();
            std::vector<char *> &vertex_scratch_ref = queue_vertex_batch.front();
            _gstore->get_vertex_batch_wait(tid); // wait for Asyn I/O

            // Compute distance(remote)
            for (unsigned m = 0; m < item_scratch_ref.size(); ++m)
            {
                item_t m_item = item_scratch_ref[m];
                if (m_item.first == _gstore->sid)
                    continue;
                float *base = (float *)(vertex_scratch_ref[m] + _data_offset);
                float dist = _distance->compare(base, query, (unsigned)_dimension);
                bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
                if (succ)
                    _gstore->update_thd_cache(tid, m_item, vertex_scratch_ref[m]); // for cache
            }
            queue_item_batch.pop_front();
            queue_vertex_batch.pop_front();
        }

        for (size_t i = 0; i < K; i++)
        {
            indices[i] = *((unsigned *)(_gstore->get_vertex(tid, best_L_nodes[i].item) + _label_offset));
        }
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        // stats->total_us += (float)duration / 1e3;
    }

    /* 使用 std::vector<hop> 代替queue */
    // void Index::search_base_index_distributed(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices, item_t ep, int relax)
    // {
    //     if (relax > _gstore->max_relax)
    //         throw std::runtime_error("error: relax > _gstore->max_relax");

    //     dsmann::InMemQueryScratch *scratch = _query_scratch_distributed[tid];
    //     scratch->clear();

    //     dsmann::NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    //     std::vector<item_t> &item_scratch = scratch->item_scratch();
    //     std::vector<char *> &vertex_scratch = scratch->vertex_scratch();
    //     std::vector<tsl::robin_set<local_id_t>> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();

    //     auto is_not_visited = [this, &inserted_into_pool_rs](const item_t &item)
    //     {
    //         return inserted_into_pool_rs[item.first].find(item.second) == inserted_into_pool_rs[item.first].end();
    //     };

    //     auto insert_into_visited = [this, &inserted_into_pool_rs](const item_t &item)
    //     {
    //         inserted_into_pool_rs[item.first].insert(item.second);
    //     };

    //     _gstore->reset_thd_ctx(tid);

    //     std::vector<item_t> init_ids;
    //     init_ids.push_back(ep);

    //     // 依赖松弛需要至少 relax + 1 个ep（todo：之后再完善）
    //     for (size_t i = 0; i < relax; i++) // 多入口会使得qps升高一些
    //     {
    //         ep.second++;
    //         init_ids.push_back(ep);
    //     }

    //     // traverse graph
    //     for (const item_t &item : init_ids)
    //     {
    //         if (is_not_visited(item))
    //         {
    //             insert_into_visited(item);
    //             item_scratch.push_back(item);
    //         }
    //     }
    //     // Read vector
    //     _gstore->get_vertex_batch(tid, item_scratch.data(), item_scratch.size(), vertex_scratch);
    //     // compute distance
    //     for (size_t i = 0; i < item_scratch.size(); ++i)
    //     {
    //         item_t item = item_scratch[i];
    //         float *base = (float *)(vertex_scratch[i] + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         bool succ = best_L_nodes.insert(dsmann::Neighbor(item, dist));
    //         if (succ)
    //             _gstore->update_thd_cache(tid, item, vertex_scratch[i]);
    //     }
    //     item_scratch.clear();
    //     vertex_scratch.clear();

    //     // 使用 std::vector<hop>
    //     std::vector<hop> &hops = _gstore->thd_hops[tid];

    //     int hid = 0;
    //     int delay = 0;
    //     while (best_L_nodes.has_unexpanded_node())
    //     {
    //         std::vector<item_t> &item_scratch_write = hops[hid].item_batch;
    //         std::vector<char *> &vertex_scratch_write = hops[hid].vertex_batch;
    //         std::vector<int> &polls = hops[hid].polls;

    //         item_scratch_write.clear();
    //         vertex_scratch_write.clear();

    //         item_t n_item = best_L_nodes.closest_unexpanded().item;

    //         // traverse graph
    //         char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
    //         unsigned MaxM = *((unsigned *)tmp);
    //         tmp += sizeof(unsigned);
    //         for (unsigned m = 0; m < MaxM; ++m)
    //         {
    //             item_t m_item;
    //             m_item.first = *((server_id_t *)tmp);
    //             tmp += sizeof(server_id_t);
    //             m_item.second = *((local_id_t *)tmp);
    //             tmp += sizeof(local_id_t);
    //             if (is_not_visited(m_item))
    //             {
    //                 insert_into_visited(m_item);
    //                 item_scratch_write.push_back(m_item);
    //             }
    //         }
    //         // Read vector
    //         _gstore->get_vertex_batch_hop(tid, item_scratch_write.data(), item_scratch_write.size(), vertex_scratch_write, polls); // for Asyn I/O
    //         // Compute distance(local)
    //         for (unsigned m = 0; m < item_scratch_write.size(); ++m)
    //         {
    //             item_t m_item = item_scratch_write[m];
    //             if (m_item.first != _gstore->sid)
    //                 continue;
    //             float *base = (float *)(vertex_scratch_write[m] + _data_offset);
    //             float dist = _distance->compare(base, query, (unsigned)_dimension);
    //             bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
    //             if (succ)
    //                 _gstore->update_thd_cache(tid, m_item, vertex_scratch_write[m]); // for cache
    //         }

    //         hid = (hid + 1) % (relax + 1);
    //         if (delay < relax)
    //         {
    //             delay++;
    //             continue;
    //         }
    //         const std::vector<item_t> &item_scratch_ref = hops[hid].item_batch;
    //         const std::vector<char *> &vertex_scratch_ref = hops[hid].vertex_batch;
    //         std::vector<int> &polls_ref = hops[hid].polls;
    //         _gstore->get_vertex_batch_wait_hop(tid, polls_ref); // wait for Asyn I/O

    //         // Compute distance(remote)
    //         for (unsigned m = 0; m < item_scratch_ref.size(); ++m)
    //         {
    //             item_t m_item = item_scratch_ref[m];
    //             if (m_item.first == _gstore->sid)
    //                 continue;
    //             float *base = (float *)(vertex_scratch_ref[m] + _data_offset);
    //             float dist = _distance->compare(base, query, (unsigned)_dimension);
    //             bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
    //             if (succ)
    //                 _gstore->update_thd_cache(tid, m_item, vertex_scratch_ref[m]); // for cache
    //         }
    //     }

    //     /* 处理队列中剩余的vertex */
    //     while (delay > 0)
    //     {
    //         hid = (hid + 1) % (relax + 1);
    //         delay--;
    //         const std::vector<item_t> &item_scratch_ref = hops[hid].item_batch;
    //         const std::vector<char *> &vertex_scratch_ref = hops[hid].vertex_batch;
    //         std::vector<int> &polls = hops[hid].polls;
    //         _gstore->get_vertex_batch_wait_hop(tid, polls); // wait for Asyn I/O

    //         // Compute distance(remote)
    //         for (unsigned m = 0; m < item_scratch_ref.size(); ++m)
    //         {
    //             item_t m_item = item_scratch_ref[m];
    //             if (m_item.first == _gstore->sid)
    //                 continue;
    //             float *base = (float *)(vertex_scratch_ref[m] + _data_offset);
    //             float dist = _distance->compare(base, query, (unsigned)_dimension);
    //             bool succ = best_L_nodes.insert(dsmann::Neighbor(m_item, dist));
    //             if (succ)
    //                 _gstore->update_thd_cache(tid, m_item, vertex_scratch_ref[m]); // for cache
    //         }
    //     }

    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, best_L_nodes[i].item) + _label_offset));
    //     }
    // }

    /*
     * 通过 get_vertex_batch 来进行邻居扩展
     * 只松弛远程访问
     */
    // void Index::search_base_index_distributed(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices, item_t ep, int relax)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销）

    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     // 依赖松弛需要至少 relax + 1 个ep
    //     for (size_t i = 0; i < relax; i++) // 多入口会使得qps升高一些
    //     {
    //         ep.second++;
    //         init_ids[tmp_l] = ep;
    //         tmp_l++;
    //         id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //         flags[id] = true;
    //     }

    //     _gstore->reset_thd_ctx(tid);
    //     std::vector<item_t> key_batch;
    //     std::vector<char *> vertex_batch;
    //     key_batch.reserve(_base_graph_R);
    //     vertex_batch.reserve(_base_graph_R);

    //     std::queue<std::vector<item_t>> queue_key_batch;
    //     std::queue<std::vector<char *>> queue_vertex_batch;

    //     // Read vector
    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         key_batch.push_back(init_ids[i]);
    //     }
    //     ASSERT(L < _base_graph_R);
    //     _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch);
    //     // compute distance
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = key_batch[i];
    //         float *base = (float *)(vertex_batch[i] + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //         _gstore->update_thd_cache(tid, item, vertex_batch[i]);
    //     }
    //     key_batch.clear();
    //     vertex_batch.clear();

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;
    //     while (k < (int)L)
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;
    //             // Read graph
    //             char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
    //             // Read vector
    //             key_batch.clear();
    //             vertex_batch.clear();
    //             unsigned MaxM = *((unsigned *)tmp);
    //             tmp += sizeof(unsigned);
    //             item_t m_item;
    //             for (unsigned m = 0; m < MaxM; ++m)
    //             {
    //                 m_item.first = *((server_id_t *)tmp);
    //                 tmp += sizeof(server_id_t);
    //                 m_item.second = *((local_id_t *)tmp);
    //                 tmp += sizeof(local_id_t);
    //                 unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                 if (flags[id])
    //                     continue;
    //                 flags[id] = 1;

    //                 key_batch.push_back(m_item);
    //             }
    //             // _gstore->get_vertex_batch_merged(tid, key_batch.data(), key_batch.size(), vertex_batch, true); // for Asyn I/O
    //             _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch, true); // for Asyn I/O
    //             queue_key_batch.push(key_batch);
    //             queue_vertex_batch.push(vertex_batch);

    //             // Compute distance(local)
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 if (m_item.first != _gstore->sid)
    //                     continue;
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // for cache
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }

    //             if (relax > 0)
    //             {
    //                 relax--;
    //                 continue;
    //             }
    //             std::vector<item_t> &key_batch_ref = queue_key_batch.front();
    //             std::vector<char *> &vertex_batch_ref = queue_vertex_batch.front();
    //             _gstore->get_vertex_batch_wait(tid); // wait for Asyn I/O

    //             // Compute distance(remote)
    //             for (unsigned m = 0; m < key_batch_ref.size(); ++m)
    //             {
    //                 m_item = key_batch_ref[m];
    //                 if (m_item.first == _gstore->sid)
    //                     continue;
    //                 float *base = (float *)(vertex_batch_ref[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch_ref[m]); // for cache
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }
    //             queue_key_batch.pop();
    //             queue_vertex_batch.pop();
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     /* 处理队列中剩余的vertex */
    //     while (!queue_key_batch.empty())
    //     {
    //         std::vector<item_t> &key_batch_ref = queue_key_batch.front();
    //         std::vector<char *> &vertex_batch_ref = queue_vertex_batch.front();
    //         _gstore->get_vertex_batch_wait(tid);
    //         item_t m_item;
    //         // Compute distance(remote)
    //         for (unsigned m = 0; m < key_batch_ref.size(); ++m)
    //         {
    //             m_item = key_batch_ref[m];
    //             if (m_item.first == _gstore->sid)
    //                 continue;
    //             float *base = (float *)(vertex_batch_ref[m] + _data_offset);
    //             float dist = _distance->compare(base, query, (unsigned)_dimension);
    //             // if (dist >= retset[L - 1].distance) // 修改这行
    //             if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                 continue;
    //             NeighborItem nn(m_item, dist, true);
    //             int r = InsertIntoPoolItem(retset.data(), L, nn);
    //             _gstore->update_thd_cache(tid, m_item, vertex_batch_ref[m]); // for cache
    //             if (L + 1 < retset.size())
    //                 ++L; // 加上这行
    //         }
    //         queue_key_batch.pop();
    //         queue_vertex_batch.pop();
    //     }
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //     }
    // }

    /*
     * 有性能损失
     * get_vertex 函数调用开销较大
     * buf_cpy new 操作和 std::memcpy 开销较小（之后优化）
     * boost::dynamic_bitset 有大概 100us 的初始化开销
     */
    // void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销，时间开销50-100us）

    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
    //     item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = init_ids[i];
    //         float *base = (float *)(_gstore->get_vertex(tid, item) + _data_offset);
    //         // char *ptr = (item.first == 0 ? _gstore->mem->kvstore() : _gstore->rdma_cache_data) + (uint64_t)item.second * _element_size;
    //         // float *base = (float *)(ptr + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //     }

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;
    //     unique_ptr<char[]> buf_cpy(new char[_element_size]);
    //     while (k < (int)L)
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;

    //             // char *tmp = (char *)(_gstore->get_vertex(tid, n_item) + _neighbor_offset); // 出bug，因为buf的内容会被修改，每次调用时上次调用的buf内容将会失效，因此需要复制一份新的
    //             std::memcpy(buf_cpy.get(), _gstore->get_vertex(tid, n_item), _element_size); // memcpy 性能开销有多大
    //             char *tmp = buf_cpy.get() + _neighbor_offset;
    //             // char *ptr = (n_item.first == 0 ? _gstore->mem->kvstore() : _gstore->rdma_cache_data) + (uint64_t)n_item.second * _element_size;
    //             // char *tmp = ptr + _neighbor_offset;
    //             unsigned MaxM = *((unsigned *)tmp);
    //             tmp += sizeof(unsigned);
    //             item_t m_item;
    //             for (unsigned m = 0; m < MaxM; ++m)
    //             {
    //                 m_item.first = *((server_id_t *)tmp);
    //                 tmp += sizeof(server_id_t);
    //                 m_item.second = *((local_id_t *)tmp);
    //                 tmp += sizeof(local_id_t);
    //                 unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                 if (flags[id])
    //                     continue;
    //                 flags[id] = 1;
    //                 float *base = (float *)(_gstore->get_vertex(tid, m_item) + _data_offset);
    //                 // char *ptr = (m_item.first == 0 ? _gstore->mem->kvstore() : _gstore->rdma_cache_data) + (uint64_t)m_item.second * _element_size;
    //                 // float *base = (float *)(ptr + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //         // char *ptr = (retset[i].item.first == 0 ? _gstore->mem->kvstore() : _gstore->rdma_cache_data) + (uint64_t)retset[i].item.second * _element_size;
    //         // indices[i] = *((unsigned *)(ptr + _label_offset));
    //     }
    // }

    /*
     * 通过 get_vertex_batch 来进行邻居扩展
     * 使用cache保存访问过的vertex数据，check读图数据时查cache，qps790
     */
    // void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销）

    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
    //     item_t ep = std::make_pair((server_id_t)1, (local_id_t)2960); // 临时赋值(remote_neighbor_order)
    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     _gstore->set_thd_buf_blk(tid, _base_graph_R * _element_size, 1);
    //     _gstore->clean_thd_cache(tid);
    //     std::vector<item_t> key_batch;    // for RDMA Read input
    //     std::vector<char *> vertex_batch; // for RDMA Read output
    //     key_batch.reserve(_base_graph_R);
    //     vertex_batch.reserve(_base_graph_R);

    //     // Read vector
    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         key_batch.push_back(init_ids[i]);
    //     }
    //     ASSERT(L < _base_graph_R);
    //     _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch);
    //     // compute distance
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = key_batch[i];
    //         float *base = (float *)(vertex_batch[i] + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //         _gstore->update_thd_cache(tid, item, vertex_batch[i]);
    //     }
    //     key_batch.clear();
    //     vertex_batch.clear();

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;
    //     // auto start = std::chrono::high_resolution_clock::now();
    //     // uint64_t total_local = 0, total_remote = 0, hops = 0;
    //     while (k < (int)L)
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;
    //             // Read graph
    //             char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
    //             // Read vector
    //             key_batch.clear();
    //             vertex_batch.clear();
    //             unsigned MaxM = *((unsigned *)tmp);
    //             tmp += sizeof(unsigned);
    //             item_t m_item;
    //             for (unsigned m = 0; m < MaxM; ++m)
    //             {
    //                 m_item.first = *((server_id_t *)tmp);
    //                 tmp += sizeof(server_id_t);
    //                 m_item.second = *((local_id_t *)tmp);
    //                 tmp += sizeof(local_id_t);
    //                 unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                 if (flags[id])
    //                     continue;
    //                 flags[id] = 1;

    //                 key_batch.push_back(m_item);
    //             }
    //             _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch);

    //             // Compute distance
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // 注意：只有入队的点才需要cache，因为没有入队的点之后都不会再用到
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }

    //             // uint64_t local = 0, remote = 0;
    //             // for (unsigned m = 0; m < key_batch.size(); ++m)
    //             // {
    //             //     m_item = key_batch[m];
    //             //     if (m_item.first == _gstore->sid)
    //             //         local++;
    //             //     else
    //             //         remote++;
    //             // }
    //             // total_local += local;
    //             // total_remote += remote;
    //             // hops++;
    //             // std::cout << "hop: " << hops << ", local: " << local << ", remote: " << remote << ", sum: " << local + remote << std::endl;
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //     }
    //     // auto end = std::chrono::high_resolution_clock::now();
    //     // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //     // std::cout << "hops: " << hops << ", duration: " << duration / 1000 << ", total_local: " << total_local << ", total_remote: " << total_remote << ", sum: " << total_local + total_remote << std::endl;
    // }

    /*
     * 通过 get_vertex_batch 来进行邻居扩展
     * 考虑调整一次check中的距离计算次序，例如，在check一个点时，先计算本地的点（同时进行远程点数据的读取），再polling远程的点，大概提升5%性能
     */
    // void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销）

    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
    //     item_t ep = std::make_pair((server_id_t)1, (local_id_t)2960); // 临时赋值(remote_neighbor_order)
    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     _gstore->set_thd_buf_blk(tid, _base_graph_R * _element_size, 1);
    //     _gstore->clean_thd_cache(tid);
    //     // 用于设置 RDMA Read 输入
    //     std::vector<item_t> key_batch;
    //     // 用于保存 RDMA Read 输出
    //     std::vector<char *> vertex_batch;
    //     // 避免内存动态分配
    //     key_batch.reserve(_base_graph_R);
    //     vertex_batch.reserve(_base_graph_R);

    //     // Read vector
    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         key_batch.push_back(init_ids[i]);
    //     }
    //     ASSERT(L < _base_graph_R);
    //     _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch);
    //     // compute distance
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = key_batch[i];
    //         float *base = (float *)(vertex_batch[i] + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //         _gstore->update_thd_cache(tid, item, vertex_batch[i]);
    //     }
    //     key_batch.clear();
    //     vertex_batch.clear();

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;
    //     while (k < (int)L)
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;
    //             // Read graph
    //             char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
    //             // Read vector
    //             key_batch.clear();
    //             vertex_batch.clear();
    //             unsigned MaxM = *((unsigned *)tmp);
    //             tmp += sizeof(unsigned);
    //             item_t m_item;
    //             for (unsigned m = 0; m < MaxM; ++m)
    //             {
    //                 m_item.first = *((server_id_t *)tmp);
    //                 tmp += sizeof(server_id_t);
    //                 m_item.second = *((local_id_t *)tmp);
    //                 tmp += sizeof(local_id_t);
    //                 unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                 if (flags[id])
    //                     continue;
    //                 flags[id] = 1;

    //                 key_batch.push_back(m_item);
    //             }
    //             _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch, true); // for Asyn I/O

    //             // Compute distance(local)
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 if ((int)m_item.first != _gstore->sid)
    //                     continue;
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // 注意：只有入队的点才需要cache，因为没有入队的点之后都不会再用到
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }
    //             // Compute distance(remote)
    //             _gstore->get_vertex_batch_wait(tid); // for Asyn I/O
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 if ((int)m_item.first == _gstore->sid)
    //                     continue;
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // 注意：只有入队的点才需要cache，因为没有入队的点之后都不会再用到
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     // 这里也可以用 get_vertex_batch 来并发读
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //     }
    // }

    /*
     * 通过 get_vertex_batch 来进行邻居扩展
     * 通过预测来实现 prefetch
     */
    // void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销）

    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
    //     item_t ep = std::make_pair((server_id_t)1, (local_id_t)2960); // 临时赋值(remote_neighbor_order)
    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = init_ids[i];
    //         float *base = (float *)(_gstore->get_vertex(tid, item) + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //     }

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;

    //     _gstore->set_thd_buf_blk(tid, _base_graph_R * _element_size, 1);

    //     unique_ptr<char[]> buf_cpy(new char[_element_size]), buf_prefetched(new char[_element_size]);
    //     /* 用vector会比用数组 new ikey_t[] 慢吗 */
    //     std::vector<ikey_t> key_batch, key_batch_prefetched;       // 用于 RDMA Read 的输入
    //     std::vector<char *> vertex_batch, vertex_batch_prefetched; // 用于保存 RDMA Read 的结果
    //     /* 避免vector内存动态分配 */
    //     key_batch.reserve(_base_graph_R);
    //     key_batch_prefetched.reserve(_base_graph_R);
    //     vertex_batch.reserve(_base_graph_R);
    //     vertex_batch_prefetched.reserve(_base_graph_R);
    //     /*** for prefetch ***/
    //     bool prefetched = false;
    //     ikey_t key_prefetched;

    //     // int hit_cnt = 0, miss_cnt = 0, not_prefetched_cnt = 0;
    //     // auto start = std::chrono::high_resolution_clock::now();
    //     while (k < (int)L) // 1250 us
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;
    //             // Read Graph
    //             // if (prefetched)
    //             // {
    //             //     if (key_prefetched == n_item)
    //             //         hit_cnt++;
    //             //     if (key_prefetched != n_item)
    //             //         miss_cnt++;
    //             // }
    //             // else
    //             // {
    //             //     not_prefetched_cnt++;
    //             // }
    //             if (prefetched and key_prefetched == n_item) // already prefetched
    //                 std::memcpy(buf_cpy.get(), buf_prefetched.get(), _element_size);
    //             else
    //                 std::memcpy(buf_cpy.get(), _gstore->get_vertex(tid, n_item), _element_size); // remote acces 2.5 us
    //             // Read vector
    //             if (prefetched and key_prefetched == n_item) // already prefetched
    //             {
    //                 key_batch = key_batch_prefetched;
    //                 vertex_batch = vertex_batch_prefetched;
    //             }
    //             else
    //             {
    //                 key_batch.clear();
    //                 vertex_batch.clear();
    //                 char *tmp = buf_cpy.get() + _neighbor_offset;
    //                 unsigned MaxM = *((unsigned *)tmp);
    //                 tmp += sizeof(unsigned);
    //                 item_t m_item;
    //                 for (unsigned m = 0; m < MaxM; ++m)
    //                 {
    //                     m_item.first = *((server_id_t *)tmp);
    //                     tmp += sizeof(server_id_t);
    //                     m_item.second = *((local_id_t *)tmp);
    //                     tmp += sizeof(local_id_t);
    //                     unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                     if (flags[id])
    //                         continue;
    //                     flags[id] = 1;

    //                     key_batch.push_back(m_item);
    //                 }
    //                 _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch); // remote acces 5-10 us
    //             }

    //             // try to prefetch
    //             prefetched = false;
    //             // if (false)
    //             if (k >= 50) // when to prefetch, set k >= 0? k >= 50?
    //             {
    //                 for (int k_prdecit = k + 1; k_prdecit < (int)L; k_prdecit++)
    //                 {
    //                     if (retset[k_prdecit].flag)
    //                     {
    //                         prefetched = true;
    //                         key_prefetched = retset[k_prdecit].item;
    //                         std::memcpy(buf_prefetched.get(), _gstore->get_vertex(tid, key_prefetched), _element_size); // remote acces 2.5 us
    //                         break;
    //                     }
    //                 }
    //             }
    //             if (prefetched) // for Asyn I/O
    //             {
    //                 key_batch_prefetched.clear();
    //                 vertex_batch_prefetched.clear();
    //                 char *tmp = buf_prefetched.get() + _neighbor_offset;
    //                 unsigned MaxM = *((unsigned *)tmp);
    //                 tmp += sizeof(unsigned);
    //                 item_t m_item;
    //                 for (unsigned m = 0; m < MaxM; ++m)
    //                 {
    //                     m_item.first = *((server_id_t *)tmp);
    //                     tmp += sizeof(server_id_t);
    //                     m_item.second = *((local_id_t *)tmp);
    //                     tmp += sizeof(local_id_t);
    //                     unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                     if (flags[id])
    //                         continue;

    //                     key_batch_prefetched.push_back(m_item);
    //                 }
    //                 vertex_batch_prefetched.clear();
    //                 _gstore->get_vertex_batch(tid, key_batch_prefetched.data(), key_batch_prefetched.size(), vertex_batch_prefetched, true);
    //             }

    //             // compute distance 2-5 us
    //             item_t m_item;
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }

    //             if (prefetched) // wait for Asyn I/O
    //                 _gstore->get_vertex_batch_wait(tid);
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //     }
    //     // auto end = std::chrono::high_resolution_clock::now();
    //     // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //     // std::cout << "duration: " << duration << std::endl;
    //     // std::cout << "hit_cnt: " << hit_cnt << ", miss_cnt: " << miss_cnt << ", not_prefetched_cnt: " << not_prefetched_cnt << std::endl;
    //     // exit(1);
    // }

    /*
     * 通过 get_vertex_batch 来进行邻居扩展
     * 一阶依赖松弛，qps840
     */
    // void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销）

    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
    //     item_t ep = std::make_pair((server_id_t)1, (local_id_t)2960); // 临时赋值(remote_neighbor_order)
    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     // 依赖松弛需要至少2个ep
    //     for (size_t i = 0; i < 1; i++) // 多入口会使得qps升高一些
    //     {
    //         ep.second++;
    //         init_ids[tmp_l] = ep;
    //         tmp_l++;
    //         id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //         flags[id] = true;
    //     }

    //     _gstore->set_thd_buf_blk(tid, _base_graph_R * _element_size, 1);
    //     _gstore->clean_thd_cache(tid);
    //     std::vector<item_t> key_batch, key_batch_prefetched;
    //     std::vector<char *> vertex_batch, vertex_batch_prefetched;
    //     key_batch.reserve(_base_graph_R);
    //     key_batch_prefetched.reserve(_base_graph_R);
    //     vertex_batch.reserve(_base_graph_R);
    //     vertex_batch_prefetched.reserve(_base_graph_R);

    //     // Read vector
    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         key_batch.push_back(init_ids[i]);
    //     }
    //     ASSERT(L < _base_graph_R);
    //     _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch);
    //     // compute distance
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = key_batch[i];
    //         float *base = (float *)(vertex_batch[i] + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //         _gstore->update_thd_cache(tid, item, vertex_batch[i]);
    //     }
    //     key_batch.clear();
    //     vertex_batch.clear();

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;
    //     // auto start = std::chrono::high_resolution_clock::now();
    //     // uint64_t total_local = 0, total_remote = 0, hops = 0;
    //     while (k < (int)L)
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;
    //             // Read graph
    //             char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
    //             // Read vector
    //             key_batch_prefetched.clear();
    //             vertex_batch_prefetched.clear();
    //             unsigned MaxM = *((unsigned *)tmp);
    //             tmp += sizeof(unsigned);
    //             item_t m_item;
    //             for (unsigned m = 0; m < MaxM; ++m)
    //             {
    //                 m_item.first = *((server_id_t *)tmp);
    //                 tmp += sizeof(server_id_t);
    //                 m_item.second = *((local_id_t *)tmp);
    //                 tmp += sizeof(local_id_t);
    //                 unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                 if (flags[id])
    //                     continue;
    //                 flags[id] = 1;

    //                 key_batch_prefetched.push_back(m_item);
    //             }
    //             _gstore->get_vertex_batch(tid, key_batch_prefetched.data(), key_batch_prefetched.size(), vertex_batch_prefetched, true); // for Asyn I/O

    //             // Compute distance
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // for cache
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }

    //             _gstore->get_vertex_batch_wait(tid); // wait for Asyn I/O
    //             key_batch = key_batch_prefetched;
    //             vertex_batch = vertex_batch_prefetched;

    //             // uint64_t local = 0, remote = 0;
    //             // for (unsigned m = 0; m < key_batch.size(); ++m)
    //             // {
    //             //     m_item = key_batch[m];
    //             //     if (m_item.first == _gstore->sid)
    //             //         local++;
    //             //     else
    //             //         remote++;
    //             // }
    //             // total_local += local;
    //             // total_remote += remote;
    //             // hops++;
    //             // std::cout << "hop: " << hops << ", local: " << local << ", remote: " << remote << ", sum: " << local + remote << std::endl;
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //     }
    //     // auto end = std::chrono::high_resolution_clock::now();
    //     // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //     // std::cout << "hops: " << hops << ", duration: " << duration / 1000 << ", total_local: " << total_local << ", total_remote: " << total_remote << ", sum: " << total_local + total_remote << std::endl;
    // }

    /*
     * 通过 get_vertex_batch 来进行邻居扩展
     * 高阶松弛
     */
    // void Index::search_base_index(int tid, const float *query, size_t K, const Parameters &parameters, unsigned *indices)
    // {
    //     unsigned L = parameters.Get<unsigned>("L_search");

    //     std::vector<NeighborItem> retset(L + 1);
    //     std::vector<item_t> init_ids(L);
    //     int data_count[2];
    //     data_count[0] = 4982175;
    //     boost::dynamic_bitset<> flags{_element_num, 0}; // 临时赋值（注意：分配flags的内存有较大性能开销）

    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)3286336); // 临时赋值
    //     // item_t ep = std::make_pair((server_id_t)1, (local_id_t)2285439); // 临时赋值(gorder)
    //     item_t ep = std::make_pair((server_id_t)1, (local_id_t)2960); // 临时赋值(remote_neighbor_order)
    //     unsigned tmp_l = 0;
    //     init_ids[tmp_l] = ep;
    //     tmp_l++;
    //     unsigned id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //     flags[id] = true;

    //     // 依赖松弛需要至少 relax + 1 个ep
    //     int relax = 2;
    //     for (size_t i = 0; i < relax; i++) // 多入口会使得qps升高一些
    //     {
    //         ep.second++;
    //         init_ids[tmp_l] = ep;
    //         tmp_l++;
    //         id = (ep.first == 0 ? ep.second : data_count[0] + ep.second); // 临时赋值
    //         flags[id] = true;
    //     }

    //     _gstore->set_thd_buf_blk(tid, _base_graph_R * _element_size, 1);
    //     _gstore->clean_thd_cache(tid);
    //     std::vector<item_t> key_batch, key_batch_prefetched;
    //     std::vector<char *> vertex_batch, vertex_batch_prefetched;
    //     key_batch.reserve(_base_graph_R);
    //     key_batch_prefetched.reserve(_base_graph_R);
    //     vertex_batch.reserve(_base_graph_R);
    //     vertex_batch_prefetched.reserve(_base_graph_R);

    //     std::queue<std::vector<item_t>> queue_key_batch;
    //     std::queue<std::vector<char *>> queue_vertex_batch;

    //     // Read vector
    //     L = tmp_l;
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         key_batch.push_back(init_ids[i]);
    //     }
    //     ASSERT(L < _base_graph_R);
    //     _gstore->get_vertex_batch(tid, key_batch.data(), key_batch.size(), vertex_batch);
    //     // compute distance
    //     for (unsigned i = 0; i < L; i++)
    //     {
    //         item_t item = key_batch[i];
    //         float *base = (float *)(vertex_batch[i] + _data_offset);
    //         float dist = _distance->compare(base, query, (unsigned)_dimension);
    //         retset[i] = NeighborItem(item, dist, true);
    //         _gstore->update_thd_cache(tid, item, vertex_batch[i]);
    //     }
    //     key_batch.clear();
    //     vertex_batch.clear();

    //     std::sort(retset.begin(), retset.begin() + L);
    //     int k = 0;
    //     // auto start = std::chrono::high_resolution_clock::now();
    //     // uint64_t total_local = 0, total_remote = 0, hops = 0;
    //     while (k < (int)L)
    //     {
    //         int nk = L;

    //         if (retset[k].flag)
    //         {
    //             retset[k].flag = false;
    //             item_t n_item = retset[k].item;
    //             // Read graph
    //             char *tmp = _gstore->get_vertex(tid, n_item) + _neighbor_offset;
    //             // Read vector
    //             key_batch_prefetched.clear();
    //             vertex_batch_prefetched.clear();
    //             unsigned MaxM = *((unsigned *)tmp);
    //             tmp += sizeof(unsigned);
    //             item_t m_item;
    //             for (unsigned m = 0; m < MaxM; ++m)
    //             {
    //                 m_item.first = *((server_id_t *)tmp);
    //                 tmp += sizeof(server_id_t);
    //                 m_item.second = *((local_id_t *)tmp);
    //                 tmp += sizeof(local_id_t);
    //                 unsigned id = (m_item.first == 0 ? m_item.second : data_count[0] + m_item.second); // 临时赋值
    //                 if (flags[id])
    //                     continue;
    //                 flags[id] = 1;

    //                 key_batch_prefetched.push_back(m_item);
    //             }
    //             // auto before_post = std::chrono::high_resolution_clock::now();
    //             _gstore->get_vertex_batch(tid, key_batch_prefetched.data(), key_batch_prefetched.size(), vertex_batch_prefetched, true); // for Asyn I/O
    //             queue_key_batch.push(key_batch_prefetched);
    //             queue_vertex_batch.push(vertex_batch_prefetched);

    //             if (relax > 0)
    //             {
    //                 relax--;
    //                 continue;
    //             }
    //             key_batch = queue_key_batch.front();
    //             vertex_batch = queue_vertex_batch.front();
    //             queue_key_batch.pop();
    //             queue_vertex_batch.pop();

    //             _gstore->get_vertex_batch_wait(tid); // wait for Asyn I/O
    //             // auto after_poll = std::chrono::high_resolution_clock::now();
    //             // auto IOtime = std::chrono::duration_cast<std::chrono::nanoseconds>(after_poll - before_post).count();

    //             // Compute distance
    //             for (unsigned m = 0; m < key_batch.size(); ++m)
    //             {
    //                 m_item = key_batch[m];
    //                 float *base = (float *)(vertex_batch[m] + _data_offset);
    //                 float dist = _distance->compare(base, query, (unsigned)_dimension);
    //                 // if (dist >= retset[L - 1].distance) // 修改这行
    //                 if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                     continue;
    //                 NeighborItem nn(m_item, dist, true);
    //                 int r = InsertIntoPoolItem(retset.data(), L, nn);
    //                 _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // for cache
    //                 if (L + 1 < retset.size())
    //                     ++L; // 加上这行
    //                 if (r < nk)
    //                     nk = r;
    //             }

    //             // uint64_t local = 0, remote = 0;
    //             // for (unsigned m = 0; m < key_batch.size(); ++m)
    //             // {
    //             //     m_item = key_batch[m];
    //             //     if (m_item.first == _gstore->sid)
    //             //         local++;
    //             //     else
    //             //         remote++;
    //             // }
    //             // total_local += local;
    //             // total_remote += remote;
    //             // hops++;
    //             // std::cout << "hop: " << hops << ", local: " << local << ", remote: " << remote << ", sum: " << local + remote << ", I/O time: " << (double)IOtime / 1000 << " us" << std::endl;
    //         }
    //         if (nk <= k)
    //             k = nk;
    //         else
    //             ++k;
    //     }
    //     /* 处理队列中剩余的vertex */
    //     while (!queue_key_batch.empty())
    //     {
    //         key_batch = queue_key_batch.front();
    //         vertex_batch = queue_vertex_batch.front();
    //         queue_key_batch.pop();
    //         queue_vertex_batch.pop();
    //         _gstore->get_vertex_batch_wait(tid);
    //         item_t m_item;
    //         for (unsigned m = 0; m < key_batch.size(); ++m)
    //         {
    //             m_item = key_batch[m];
    //             float *base = (float *)(vertex_batch[m] + _data_offset);
    //             float dist = _distance->compare(base, query, (unsigned)_dimension);
    //             // if (dist >= retset[L - 1].distance) // 修改这行
    //             if (L + 1 == retset.size() and dist >= retset[L - 1].distance)
    //                 continue;
    //             NeighborItem nn(m_item, dist, true);
    //             int r = InsertIntoPoolItem(retset.data(), L, nn);
    //             _gstore->update_thd_cache(tid, m_item, vertex_batch[m]); // for cache
    //             if (L + 1 < retset.size())
    //                 ++L; // 加上这行
    //         }
    //         // uint64_t local = 0, remote = 0;
    //         // for (unsigned m = 0; m < key_batch.size(); ++m)
    //         // {
    //         //     m_item = key_batch[m];
    //         //     if (m_item.first == _gstore->sid)
    //         //         local++;
    //         //     else
    //         //         remote++;
    //         // }
    //         // total_local += local;
    //         // total_remote += remote;
    //         // hops++;
    //         // std::cout << "hop: " << hops << ", local: " << local << ", remote: " << remote << ", sum: " << local + remote << std::endl;
    //     }
    //     for (size_t i = 0; i < K; i++)
    //     {
    //         indices[i] = *((unsigned *)(_gstore->get_vertex(tid, retset[i].item) + _label_offset));
    //     }
    //     // auto end = std::chrono::high_resolution_clock::now();
    //     // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //     // std::cout << "duration: " << duration / 1000 << ", total_local: " << total_local << ", total_remote: " << total_remote << ", sum: " << total_local + total_remote << std::endl;
    //     // exit(1);
    // }

} // namespace numaann
