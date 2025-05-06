#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <vector>
#include <mutex>
#include <random>
#include "util.h"
#include "tsl/robin_set.h"

namespace numaann
{
    /* use for RDMA-ANNS*/
    struct NeighborItem
    {
        item_t item;
        float distance;
        bool flag;
        NeighborItem() = default;
        NeighborItem(item_t item, float distance, bool f) : item{item}, distance{distance}, flag(f) {}

        inline bool operator<(const NeighborItem &other) const
        {
            return distance < other.distance;
        }
    };

    /* use for RDMA-ANNS*/
    static inline int InsertIntoPoolItem(NeighborItem *addr, unsigned K, NeighborItem nn)
    {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].distance > nn.distance)
        {
            memmove((char *)&addr[left + 1], &addr[left], K * sizeof(NeighborItem));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance)
        {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)
                right = mid;
            else
                left = mid;
        }
        // check equal ID

        while (left > 0)
        {
            if (addr[left].distance < nn.distance)
                break;
            if (addr[left].item == nn.item)
                return K + 1;
            left--;
        }
        if (addr[left].item == nn.item || addr[right].item == nn.item)
            return K + 1;
        memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(NeighborItem));
        addr[right] = nn;
        return right;
    }

}

namespace numaann
{

    struct Neighbor
    {
        unsigned id;
        float distance;
        bool flag;

        Neighbor() = default;
        Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

        inline bool operator<(const Neighbor &other) const
        {
            return distance < other.distance;
        }
    };

    typedef std::lock_guard<std::mutex> LockGuard;
    struct nhood
    {
        std::mutex lock;
        std::vector<Neighbor> pool;
        unsigned M;

        std::vector<unsigned> nn_old;
        std::vector<unsigned> nn_new;
        std::vector<unsigned> rnn_old;
        std::vector<unsigned> rnn_new;

        nhood() {}
        nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N)
        {
            M = s;
            nn_new.resize(s * 2);
            GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N);
            nn_new.reserve(s * 2);
            pool.reserve(l);
        }

        nhood(const nhood &other)
        {
            M = other.M;
            std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
        }
        void insert(unsigned id, float dist)
        {
            LockGuard guard(lock);
            if (dist > pool.front().distance)
                return;
            for (unsigned i = 0; i < pool.size(); i++)
            {
                if (id == pool[i].id)
                    return;
            }
            if (pool.size() < pool.capacity())
            {
                pool.push_back(Neighbor(id, dist, true));
                std::push_heap(pool.begin(), pool.end());
            }
            else
            {
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            }
        }

        template <typename C>
        void join(C callback) const
        {
            for (unsigned const i : nn_new)
            {
                for (unsigned const j : nn_new)
                {
                    if (i < j)
                    {
                        callback(i, j);
                    }
                }
                for (unsigned j : nn_old)
                {
                    callback(i, j);
                }
            }
        }
    };

    struct SimpleNeighbor
    {
        unsigned id;
        float distance;

        SimpleNeighbor() = default;
        SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}

        inline bool operator<(const SimpleNeighbor &other) const
        {
            return distance < other.distance;
        }
    };
    struct SimpleNeighbors
    {
        std::vector<SimpleNeighbor> pool;
    };

    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
    {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].distance > nn.distance)
        {
            memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance)
        {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)
                right = mid;
            else
                left = mid;
        }
        // check equal ID

        while (left > 0)
        {
            if (addr[left].distance < nn.distance)
                break;
            if (addr[left].id == nn.id)
                return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)
            return K + 1;
        memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

}

namespace diskann
{
    struct Neighbor
    {
        unsigned id;
        float distance;
        bool expanded;

        Neighbor() = default;

        Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
        {
        }

        inline bool operator<(const Neighbor &other) const
        {
            return distance < other.distance || (distance == other.distance && id < other.id);
        }

        inline bool operator==(const Neighbor &other) const
        {
            return (id == other.id);
        }
    };

    // Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
    //            the first Neighbor which is unexpanded.
    class NeighborPriorityQueue
    {
    public:
        NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
        {
        }

        explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
        {
        }

        // Inserts the item ordered into the set up to the sets capacity.
        // The item will be dropped if it is the same id as an exiting
        // set item or it has a greated distance than the final
        // item in the set. The set cursor that is used to pop() the
        // next item will be set to the lowest index of an uncheck item
        bool insert(const Neighbor &nbr)
        {
            if (_size == _capacity && _data[_size - 1] < nbr)
            {
                return false;
            }

            size_t lo = 0, hi = _size;
            while (lo < hi)
            {
                size_t mid = (lo + hi) >> 1;
                if (nbr < _data[mid])
                {
                    hi = mid;
                    // Make sure the same id isn't inserted into the set
                }
                else if (_data[mid].id == nbr.id)
                {
                    return false;
                }
                else
                {
                    lo = mid + 1;
                }
            }

            if (lo < _capacity)
            {
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
            }
            _data[lo] = {nbr.id, nbr.distance};
            if (_size < _capacity)
            {
                _size++;
            }
            if (lo < _cur)
            {
                _cur = lo;
            }
            return true;
        }

        Neighbor closest_unexpanded()
        {
            _data[_cur].expanded = true;
            size_t pre = _cur;
            while (_cur < _size && _data[_cur].expanded)
            {
                _cur++;
            }
            return _data[pre];
        }

        bool has_unexpanded_node() const
        {
            return _cur < _size;
        }

        size_t size() const
        {
            return _size;
        }

        size_t capacity() const
        {
            return _capacity;
        }

        void reserve(size_t capacity)
        {
            if (capacity + 1 > _data.size())
            {
                _data.resize(capacity + 1);
            }
            _capacity = capacity;
        }

        Neighbor &operator[](size_t i)
        {
            return _data[i];
        }

        Neighbor operator[](size_t i) const
        {
            return _data[i];
        }

        void clear()
        {
            _size = 0;
            _cur = 0;
        }

    private:
        size_t _size, _capacity, _cur;
        std::vector<Neighbor> _data;
    };
} // namespace diskann

namespace diskann
{
    const float GRAPH_SLACK_FACTOR = 1.3f;
    class InMemQueryScratch
    {
    public:
        InMemQueryScratch(uint32_t search_l, uint32_t r)
            : _L(0), _R(r)
        {
            resize_for_new_L(search_l);
            _inserted_into_pool_bs = new boost::dynamic_bitset<>();

            _id_scratch.reserve((size_t)std::ceil(1.5 * GRAPH_SLACK_FACTOR * _R));
            _vertex_scratch.reserve((size_t)std::ceil(1.5 * GRAPH_SLACK_FACTOR * _R));
        }

        ~InMemQueryScratch()
        {
            delete _inserted_into_pool_bs;
        }

        void clear()
        {
            _best_l_nodes.clear();

            _inserted_into_pool_rs.clear();
            _inserted_into_pool_bs->reset();

            _id_scratch.clear();
            _vertex_scratch.clear();
        }

        void resize_for_new_L(uint32_t new_l)
        {
            if (new_l > _L)
            {
                _L = new_l;
                _best_l_nodes.reserve(_L);

                _inserted_into_pool_rs.reserve(20 * _L);
            }
            else
            {
                throw std::runtime_error("error@resize_for_new_L: resize failed.");
            }
        }

        inline uint32_t get_L()
        {
            return _L;
        }
        inline uint32_t get_R()
        {
            return _R;
        }
        inline NeighborPriorityQueue &best_l_nodes()
        {
            return _best_l_nodes;
        }
        inline tsl::robin_set<uint32_t> &inserted_into_pool_rs()
        {
            return _inserted_into_pool_rs;
        }
        inline boost::dynamic_bitset<> &inserted_into_pool_bs()
        {
            return *_inserted_into_pool_bs;
        }
        inline std::vector<uint32_t> &id_scratch()
        {
            return _id_scratch;
        }
        inline std::vector<char *> &vertex_scratch()
        {
            return _vertex_scratch;
        }

    private:
        uint32_t _L;
        uint32_t _R;

        // _best_l_nodes is reserved for storing best L entries
        // Underlying storage is L+1 to support inserts
        NeighborPriorityQueue _best_l_nodes;

        // Capacity initialized to 20L
        tsl::robin_set<uint32_t> _inserted_into_pool_rs;

        // Use a pointer here to allow for forward declaration of dynamic_bitset
        // in public headers to avoid making boost a dependency for clients
        // of DiskANN.
        boost::dynamic_bitset<> *_inserted_into_pool_bs;

        // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
        std::vector<uint32_t> _id_scratch;
        std::vector<char *> _vertex_scratch;
    };
} // namespace diskann

namespace dsmann
{
    struct Neighbor
    {
        item_t item;
        float distance;
        bool expanded;

        Neighbor() = default;

        Neighbor(item_t item, float distance) : item{item}, distance{distance}, expanded(false)
        {
        }

        inline bool operator<(const Neighbor &other) const
        {
            return distance < other.distance || (distance == other.distance && item < other.item);
        }

        inline bool operator==(const Neighbor &other) const
        {
            return (item == other.item);
        }
    };

    // Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
    //            the first Neighbor which is unexpanded.
    class NeighborPriorityQueue
    {
    public:
        NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
        {
        }

        explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
        {
        }

        // Inserts the item ordered into the set up to the sets capacity.
        // The item will be dropped if it is the same id as an exiting
        // set item or it has a greated distance than the final
        // item in the set. The set cursor that is used to pop() the
        // next item will be set to the lowest index of an uncheck item
        bool insert(const Neighbor &nbr)
        {
            if (_size == _capacity && _data[_size - 1] < nbr)
            {
                return false;
            }

            size_t lo = 0, hi = _size;
            while (lo < hi)
            {
                size_t mid = (lo + hi) >> 1;
                if (nbr < _data[mid])
                {
                    hi = mid;
                    // Make sure the same id isn't inserted into the set
                }
                else if (_data[mid].item == nbr.item)
                {
                    return false;
                }
                else
                {
                    lo = mid + 1;
                }
            }

            if (lo < _capacity)
            {
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
            }
            _data[lo] = {nbr.item, nbr.distance};
            if (_size < _capacity)
            {
                _size++;
            }
            if (lo < _cur)
            {
                _cur = lo;
            }
            return true;
        }

        Neighbor closest_unexpanded()
        {
            _data[_cur].expanded = true;
            size_t pre = _cur;
            while (_cur < _size && _data[_cur].expanded)
            {
                _cur++;
            }
            return _data[pre];
        }

        bool has_unexpanded_node() const
        {
            return _cur < _size;
        }

        size_t size() const
        {
            return _size;
        }

        size_t capacity() const
        {
            return _capacity;
        }

        void reserve(size_t capacity)
        {
            if (capacity + 1 > _data.size())
            {
                _data.resize(capacity + 1);
            }
            _capacity = capacity;
        }

        Neighbor &operator[](size_t i)
        {
            return _data[i];
        }

        Neighbor operator[](size_t i) const
        {
            return _data[i];
        }

        void clear()
        {
            _size = 0;
            _cur = 0;
        }

    private:
        size_t _size, _capacity, _cur;
        std::vector<Neighbor> _data;
    };
} // namespace dsmann

namespace dsmann
{
    const float GRAPH_SLACK_FACTOR = 1.3f;
    class InMemQueryScratch
    {
    public:
        InMemQueryScratch(uint32_t search_l, uint32_t r, uint32_t num_servers)
            : _L(0), _R(r)
        {
            resize_for_new_L(search_l);

            _inserted_into_pool_rs.resize(num_servers);

            _item_scratch.reserve((size_t)std::ceil(1.5 * GRAPH_SLACK_FACTOR * _R));
            _vertex_scratch.reserve((size_t)std::ceil(1.5 * GRAPH_SLACK_FACTOR * _R));
        }

        ~InMemQueryScratch()
        {
        }

        void clear()
        {
            _best_l_nodes.clear();

            for (size_t i = 0; i < _inserted_into_pool_rs.size(); i++)
            {
                _inserted_into_pool_rs[i].clear();
            }

            _item_scratch.clear();
            _vertex_scratch.clear();
        }

        void resize_for_new_L(uint32_t new_l)
        {
            if (new_l > _L)
            {
                _L = new_l;
                _best_l_nodes.reserve(_L);

                for (size_t i = 0; i < _inserted_into_pool_rs.size(); i++)
                {
                    _inserted_into_pool_rs[i].reserve(20 * _L);
                }
            }
            else
            {
                throw std::runtime_error("error@resize_for_new_L: resize failed.");
            }
        }

        inline uint32_t get_L()
        {
            return _L;
        }
        inline uint32_t get_R()
        {
            return _R;
        }
        inline NeighborPriorityQueue &best_l_nodes()
        {
            return _best_l_nodes;
        }
        inline std::vector<tsl::robin_set<local_id_t>> &inserted_into_pool_rs()
        {
            return _inserted_into_pool_rs;
        }
        inline std::vector<item_t> &item_scratch()
        {
            return _item_scratch;
        }
        inline std::vector<char *> &vertex_scratch()
        {
            return _vertex_scratch;
        }

    private:
        uint32_t _L;
        uint32_t _R;

        // _best_l_nodes is reserved for storing best L entries
        // Underlying storage is L+1 to support inserts
        NeighborPriorityQueue _best_l_nodes;

        // Capacity initialized to 20L
        std::vector<tsl::robin_set<local_id_t>> _inserted_into_pool_rs;

        // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
        std::vector<item_t> _item_scratch;
        std::vector<char *> _vertex_scratch;
    };
} // namespace dsmann

#endif // EFANNA2E_GRAPH_H
