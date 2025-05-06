/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

#pragma once

#pragma GCC diagnostic warning "-fpermissive"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>

using namespace std;

#include "global.hpp"

// utils
#include "timer.hpp"
#include "assertion.hpp"

// rdma_lib
#include "rdmaio.hpp"

using namespace rdmaio;

class RDMA {

public:
    enum MemType { CPU, GPU };

    struct MemoryRegion {
        MemType type;
        char *addr;
        uint64_t sz;
        void *mem;
    };

    class RDMA_Device {
        static const uint64_t RDMA_CTRL_PORT = 19344;
    public:
        RdmaCtrl* ctrl = NULL;

        // currently we only support one cpu and one gpu mr in mrs!
        RDMA_Device(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
            // record IPs of ndoes
            vector<string> ipset;
            ifstream ipfile(ipfn);
            string ip;

            // get first nnodes IPs
            for (int i = 0; i < nnodes; i++) {
                ipfile >> ip;
                ipset.push_back(ip);
            }

            // init device and create QPs
            ctrl = new RdmaCtrl(nid, ipset, RDMA_CTRL_PORT, true); // enable single context
            ctrl->query_devinfo();  // 查询机器上的所有 device
            ctrl->open_device();    // 打开指定的 device
            for (auto mr : mrs) {
                switch (mr.type) {
                case RDMA::MemType::CPU:
                    ctrl->set_connect_mr(mr.addr, mr.sz);
                    ctrl->register_connect_mr();
                    break;
                case RDMA::MemType::GPU:
#ifdef USE_GPU
                    ctrl->set_connect_mr_gpu(mr.addr, mr.sz);
                    ctrl->register_connect_mr_gpu();
                    break;
#else
                    logstream(LOG_ERROR) << "Build wukong w/o GPU support." << LOG_endl;
                    ASSERT(false);
#endif
                default:
                    logstream(LOG_ERROR) << "Unkown memory region." << LOG_endl;
                }
            }

            ctrl->start_server();
            for (uint j = 0; j < nthds; ++j) {
                for (uint i = 0; i < nnodes; ++i) {
                    // FIXME: statically use 1 device and 1 port
                    //
                    // devID: [0, #devs), portID: (0, #ports]
                    // 0: always choose the 1st (RDMA) device
                    // 1: always choose the 1st (RDMA) port
                    Qp *qp = ctrl->create_rc_qp(j, i, 0, 1);
                    ASSERT(qp != NULL);
                }
            }

            // connect all QPs
            while (1) {
                int connected = 0;
                for (uint j = 0; j < nthds; ++j) {
                    for (uint i = 0; i < nnodes; ++i) {
                        Qp *qp = ctrl->get_rc_qp(j, i);

                        if (qp->inited_) // has connected
                            connected ++;
                        else if (qp->connect_rc())
                            connected ++;
                    }
                }

                if (connected == nthds * nnodes) break; // done
            }

            // 输出 qp 信息
            struct ibv_qp_attr qp_attr;
            struct ibv_qp_init_attr qp_init_attr;
            int ret = ibv_query_qp(ctrl->get_rc_qp(0, 1)->qp, &qp_attr, 0, &qp_init_attr);
            if (ret)
            {
                logstream(LOG_ERROR) << "failed to query QP properties." << LOG_endl;
            }
            logger(LOG_INFO, "init max_inline_data: %u", qp_init_attr.cap.max_inline_data);
            logger(LOG_INFO, "max_inline_data: %u", qp_attr.cap.max_inline_data);
            logger(LOG_INFO, "max_rd_atomic: %u", qp_attr.max_rd_atomic);
            logger(LOG_INFO, "max_dest_rd_atomic: %u", qp_attr.max_dest_rd_atomic);

            // 若同一个QP的qp_num和dest_qp_num相同，即自环通信
            // for (uint j = 0; j < nthds; ++j) {
            //     for (uint i = 0; i < nnodes; ++i) {
            //         Qp *qp = ctrl->get_rc_qp(j, i);
            //         int ret = ibv_query_qp(qp->qp, &qp_attr, 0, &qp_init_attr);
            //         logstream(LOG_EMPH) << "#" << nid << ": qp_num = " << qp->qp->qp_num << ", dest_qp_num: " << qp_attr.dest_qp_num << LOG_endl;
            //     }
            // }
        }

#ifdef USE_GPU
        // (sync) GPUDirect RDMA Write (w/ completion)
        int GPURdmaWrite(int tid, int nid, char *local_gpu,
                         uint64_t sz, uint64_t off, bool to_gpu = false) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            int flags = IBV_SEND_SIGNALED;
            qp->rc_post_send_gpu(IBV_WR_RDMA_WRITE, local_gpu, sz, off, flags, to_gpu);
            qp->poll_completion();
            return 0;
        }
#endif

        inline int RdmaFetchAndAdd(int tid, int nid, char *local, uint64_t off, uint64_t add_value) {        
            Qp* qp = ctrl->get_rc_qp_nolock(tid, nid);
            qp->rc_post_fetch_and_add(local, off, add_value, IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
        }

        // (sync) RDMA Read (w/ completion)
        inline int RdmaRead(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp_nolock(tid, nid);

            // sweep remaining completion events (due to selective RDMA writes)
            if (!qp->first_send())
                qp->poll_completion();

            qp->rc_post_send(IBV_WR_RDMA_READ, local, sz, off, IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
        }

        inline int TestRdmaBW(int tid, int nid, char *local, uint64_t sz, uint64_t off, uint64_t iter, uint64_t depth)
        {
            logger(LOG_EMPH, "TestRdmaBW...");
            std::cout << "size: " << sz << std::endl;
            std::cout << "iter: " << iter << std::endl;
            std::cout << "depth: " << depth << std::endl;
            Qp* qp = ctrl->get_rc_qp_nolock(tid, nid);
            for (size_t i = 0; i < depth; i++)
            {
                qp->rc_post_send(IBV_WR_RDMA_READ, local, sz, off, IBV_SEND_SIGNALED);
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < iter; i++)
            {
                qp->rc_post_send(IBV_WR_RDMA_READ, local, sz, off, IBV_SEND_SIGNALED);
                qp->poll_completion();
            }
            auto end = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < depth; i++)
            {
                qp->poll_completion();
            }
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            double bandwidth = sz * iter / (double)duration;
            // 注意：在网络带宽测试中，通常使用的 "k" 是指 1000，而不是 1024
            logger(LOG_INFO, "bandwidth: %lf GBps", bandwidth);
            return 0;
        }

        /*
         * 每提交一个wr就poll一个wc(非常慢，不能并发IO，qps:184)
         */
        // inline int RdmaReadBatch(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size) {
        //     for (size_t i = 0; i < batch_size; i++)
        //     {
        //         Qp* qp = ctrl->get_rc_qp_nolock(tid, nid[i]);
        //         int flags = IBV_SEND_SIGNALED;
        //         qp->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
        //         Qp::IOStatus status = qp->poll_completion();
        //         if (status != Qp::IOStatus::IO_SUCC) {
        //             logstream(LOG_ERROR) << "Qp IOStatus is not success." << LOG_endl;
        //         }
        //     }
        //     return 0;
        // }

        /*
         * 先提交全部的wr再poll(qps:710)
         */
        // inline int RdmaReadBatch(int tid, int nid, char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size)
        // {
        //     Qp *qp = ctrl->get_rc_qp_nolock(tid, nid);
        //     for (size_t i = 0; i < batch_size; i++)
        //     {
        //         int flags = IBV_SEND_SIGNALED;
        //         // int flags = (i == batch_size - 1 ? IBV_SEND_SIGNALED : 0); // for unsignaled completion
        //         qp->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
        //     }
        //     Qp::IOStatus status = qp->poll_completions(batch_size);
        //     // Qp::IOStatus status = qp->poll_completion(); // for unsignaled completion
        //     if (status != Qp::IOStatus::IO_SUCC) {
        //         logstream(LOG_ERROR) << "Qp IOStatus is not success." << LOG_endl;
        //     }
        //     return 0;
        // }

        // 使用Doorbell后性能略微下降,是因为代码实现问题？
        // inline int RdmaReadBatchDoorbell(int tid, int nid, char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size) {
        //     Qp* qp = ctrl->get_rc_qp_nolock(tid, nid);
        //     RdmaReq reqs[MAX_DOORBELL_SIZE];
        //     for (uint i = 0; i < batch_size; i++) {
        //         reqs[i].opcode = IBV_WR_RDMA_READ;
        //         reqs[i].flags = IBV_SEND_SIGNALED;
        //         reqs[i].buf = (uint64_t)local_batch[i];
        //         reqs[i].length = sz;
        //         reqs[i].wr.rdma.remote_offset = off_batch[i];
        //     }
        //     qp->rc_post_doorbell(reqs, batch_size);
        //     Qp::IOStatus status = qp->poll_completions(batch_size);
        //     if (status != Qp::IOStatus::IO_SUCC) {
        //         logstream(LOG_ERROR) << "Qp IOStatus is not success." << LOG_endl;
        //     }
        //     return 0;
        // }

        /*
         * 先提交全部的wr再poll全部的wc(qps:710)
         */
        inline int RdmaReadBatch(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size) {
            Qp *qps[Global::num_servers];
            int sr_cnt[Global::num_servers];
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
                sr_cnt[sid] = 0;
            }
            // auto time0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < batch_size; i++)
            {
                int flags = IBV_SEND_SIGNALED;
                qps[nid[i]]->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
                sr_cnt[nid[i]]++;
            }
            // auto time1 = std::chrono::high_resolution_clock::now();
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                if (sr_cnt[sid] > 0)
                    qps[sid]->poll_completions(sr_cnt[sid]);
            }
            // auto time2 = std::chrono::high_resolution_clock::now();
            // auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();
            // auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
            // std::cout << "batch_size: " << batch_size << ", post duration: " << (double)duration1 / 1000 << ", poll duration: " << (double)duration2 / 1000 << ", total duration: " << (double)(duration1 + duration2) / 1000  << " us" << std::endl;
            return 0;
        }

        inline int RdmaReadBatchDoorbell(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size) {
            Qp *qps[Global::num_servers];
            int sr_cnt[Global::num_servers];
            RdmaReq reqs[MAX_DOORBELL_SIZE];
            // auto time0 = std::chrono::high_resolution_clock::now();
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
                int reqs_sz = 0;
                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                    {
                        reqs[reqs_sz].opcode = IBV_WR_RDMA_READ;
                        reqs[reqs_sz].flags = IBV_SEND_SIGNALED;
                        reqs[reqs_sz].buf = (uint64_t)local_batch[i];
                        reqs[reqs_sz].length = sz;
                        reqs[reqs_sz].wr.rdma.remote_offset = off_batch[i];
                        reqs_sz++;
                    }
                }
                sr_cnt[sid] = reqs_sz;
                if(sr_cnt[sid] > 0)
                    qps[sid]->rc_post_doorbell(reqs, reqs_sz);
            }
            // auto time1 = std::chrono::high_resolution_clock::now();
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                if (sr_cnt[sid] > 0)
                    qps[sid]->poll_completions(sr_cnt[sid]);
            }
            // auto time2 = std::chrono::high_resolution_clock::now();
            // auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();
            // auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
            // std::cout << "batch_size: " << batch_size << ", post duration: " << (double)duration1 / 1000 << ", poll duration: " << (double)duration2 / 1000 << ", total duration: " << (double)(duration1 + duration2) / 1000  << " us" << std::endl;
            return 0;
        }

        inline int RdmaReadBatch_Async_Send(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size, std::vector<int> &polls) {
            Qp *qps[Global::num_servers];
            for (int sid = 0; sid < Global::num_servers; sid++){
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
            }
            for (size_t i = 0; i < batch_size; i++)
            {
                int flags = IBV_SEND_SIGNALED;
                qps[nid[i]]->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
                polls[nid[i]]++;
            }
            return 0;
        }
        inline int RdmaReadBatchDoorbell_Async_Send(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size, std::vector<int> &polls) {
            Qp *qps[Global::num_servers];
            RdmaReq reqs[MAX_DOORBELL_SIZE];
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
                int reqs_sz = 0;
                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                    {
                        reqs[reqs_sz].opcode = IBV_WR_RDMA_READ;
                        reqs[reqs_sz].flags = IBV_SEND_SIGNALED;
                        reqs[reqs_sz].buf = (uint64_t)local_batch[i];
                        reqs[reqs_sz].length = sz;
                        reqs[reqs_sz].wr.rdma.remote_offset = off_batch[i];
                        reqs_sz++;
                    }
                }
                polls[sid] = reqs_sz;
                if(polls[sid] > 0)
                    qps[sid]->rc_post_doorbell(reqs, reqs_sz);
            }
            return 0;
        }
        inline int RdmaReadBatch_Async_Wait(int tid, std::vector<int> &polls) {
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                if (polls[sid] > 0)
                {
                    Qp *qp = ctrl->get_rc_qp_nolock(tid, sid);
                    qp->poll_completions(polls[sid]);
                    polls[sid] = 0;
                }
            }
            return 0;
        }

        // int _sr_cnt[2]{0};
        // inline int RdmaReadBatch_Async_Send(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size) {
        //     Qp *qps[Global::num_servers];
        //     for (int sid = 0; sid < Global::num_servers; sid++){
        //         qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
        //         _sr_cnt[sid] = 0;
        //     }
        //     for (size_t i = 0; i < batch_size; i++)
        //     {
        //         int flags = IBV_SEND_SIGNALED;
        //         qps[nid[i]]->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
        //         _sr_cnt[nid[i]]++;
        //     }
        //     return 0;
        // }
        // inline int RdmaReadBatch_Async_Poll(int tid) {
        //     for (int sid = 0; sid < Global::num_servers; sid++)
        //     {
        //         Qp *qp = ctrl->get_rc_qp_nolock(tid, sid);
        //         qp->poll_completions(_sr_cnt[sid]);
        //         _sr_cnt[sid] = 0;
        //     }
        //     return 0;
        // }

        /*
         * 先提交全部的wr再poll最后的wc（使用无信号完成）(qps:705，无信号完成没有性能提升？)
         */
        inline int RdmaReadBatchUnsignal(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size) {
            Qp *qps[Global::num_servers];
            int last_sr[Global::num_servers];
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
                last_sr[sid] = -1;
                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                        last_sr[sid] = i;
                }

                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                    {
                        int flags = (i == last_sr[sid] ? IBV_SEND_SIGNALED : 0);
                        qps[sid]->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
                    }
                }
            }
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                if (last_sr[sid] > -1)
                    qps[sid]->poll_completion();
            }
            return 0;
        }

        inline int RdmaReadBatchUnsignal_Async_Send(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size, std::vector<int> &polls) {
            Qp *qps[Global::num_servers];
            int last_sr[Global::num_servers];
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
                last_sr[sid] = -1;
                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                        last_sr[sid] = i;
                }

                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                    {
                        int flags = (i == last_sr[sid] ? IBV_SEND_SIGNALED : 0);
                        qps[sid]->rc_post_send(IBV_WR_RDMA_READ, local_batch[i], sz, off_batch[i], flags);
                        polls[sid]++;
                    }
                }
            }
            return 0;
        }
        inline int RdmaReadBatchDoorbellUnsignal_Async_Send(int tid, int nid[], char *local_batch[], uint64_t sz, uint64_t off_batch[], size_t batch_size, std::vector<int> &polls) {
            Qp *qps[Global::num_servers];
            RdmaReq reqs[MAX_DOORBELL_SIZE];
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
                int reqs_sz = 0;
                for (size_t i = 0; i < batch_size; i++)
                {
                    if (nid[i] == sid)
                    {
                        assert(reqs_sz < MAX_DOORBELL_SIZE);
                        reqs[reqs_sz].opcode = IBV_WR_RDMA_READ;
                        reqs[reqs_sz].flags = 0; // for unsignal competition
                        reqs[reqs_sz].buf = (uint64_t)local_batch[i];
                        reqs[reqs_sz].length = sz;
                        reqs[reqs_sz].wr.rdma.remote_offset = off_batch[i];
                        reqs_sz++;
                    }
                }
                polls[sid] = reqs_sz;
                if(polls[sid] > 0)
                    qps[sid]->rc_post_doorbell(reqs, reqs_sz, true);
            }
            return 0;
        }
        // 支持不同的 lenghth
        // inline int RdmaReadBatchDoorbellUnsignal_Async_Send(int tid, int nid[], char *local_batch[], uint64_t lenghth_bath[], uint64_t off_batch[], size_t batch_size, std::vector<int> &polls) {
        //     Qp *qps[Global::num_servers];
        //     RdmaReq reqs[MAX_DOORBELL_SIZE];
        //     for (int sid = 0; sid < Global::num_servers; sid++)
        //     {
        //         qps[sid] = ctrl->get_rc_qp_nolock(tid, sid);
        //         int reqs_sz = 0;
        //         for (size_t i = 0; i < batch_size; i++)
        //         {
        //             if (nid[i] == sid)
        //             {
        //                 reqs[reqs_sz].opcode = IBV_WR_RDMA_READ;
        //                 reqs[reqs_sz].flags = 0; // for unsignal competition
        //                 reqs[reqs_sz].buf = (uint64_t)local_batch[i];
        //                 reqs[reqs_sz].length = lenghth_bath[i]; 
        //                 reqs[reqs_sz].wr.rdma.remote_offset = off_batch[i];
        //                 reqs_sz++;
        //             }
        //         }
        //         polls[sid] = reqs_sz;
        //         if(polls[sid] > 0)
        //             qps[sid]->rc_post_doorbell(reqs, reqs_sz, true);
        //     }
        //     return 0;
        // }
        inline int RdmaReadBatchUnsignal_Async_Wait(int tid, std::vector<int> &polls) {
            for (int sid = 0; sid < Global::num_servers; sid++)
            {
                if (polls[sid] > 0){
                    Qp *qp = ctrl->get_rc_qp_nolock(tid, sid);
                    qp->poll_completion();
                    polls[sid] = 0;
                }
            }
            return 0;
        }

        // (sync) RDMA Write (w/ completion)
        inline int RdmaWrite(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            int flags = IBV_SEND_SIGNALED;
            qp->rc_post_send(IBV_WR_RDMA_WRITE, local, sz, off, flags);
            qp->poll_completion();
            return 0;
        }

        // (blind) RDMA Write (w/o completion)
        inline int RdmaWriteNonSignal(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);
            int flags = 0;
            qp->rc_post_send(IBV_WR_RDMA_WRITE, local, sz, off, flags);
            return 0;
        }

        // (adaptive) RDMA Write (w/o completion)
        inline int RdmaWriteSelective(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            int flags = (qp->first_send() ? IBV_SEND_SIGNALED : 0);
            qp->rc_post_send(IBV_WR_RDMA_WRITE, local, sz, off, flags);
            if (qp->need_poll())  // sweep all completion (batch)
                qp->poll_completion();
            return 0;
        }
    };

    RDMA_Device *dev = NULL;

    RDMA() { }

    ~RDMA() { if (dev != NULL) delete dev; }

    void init_dev(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
        dev = new RDMA_Device(nnodes, nthds, nid, mrs, ipfn);
    }

    inline static bool has_rdma() { return true; }

    inline static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
};

inline void RDMA_init(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
    uint64_t t = timer::get_usec();

    // init RDMA device
    RDMA &rdma = RDMA::get_rdma();
    rdma.init_dev(nnodes, nthds, nid, mrs, ipfn);

    t = timer::get_usec() - t;
    logstream(LOG_INFO) << "initializing RMDA done (" << t / 1000 << " ms)" << LOG_endl;
}
