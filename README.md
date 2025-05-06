# RED-ANNS

### 机器配置

安装依赖

```shell
sudo apt install numactl
sudo apt-get install libboost-all-dev
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

下载 boost_1_85_0.tar.gz https://www.boost.org/users/history/version_1_85_0.html

```shell
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
tar -xzvf boost_1_85_0.tar.gz
cd boost_1_85_0
./bootstrap.sh --prefix=/usr/
./bootstrap.sh --prefix=/usr/local/
sudo ./b2 install
```

配置当前shell所在机器到其他机器的ssh免密登录(包括自己到自己到免密登录)，方便使用rsync

解除所有机器的最大锁定内存限制

配置网络接口的MTU

### 项目配置

在 hosts 文件中设置 server ip 地址（注意末尾也要有换行）

```c
10.176.24.160
10.176.24.162

```

在 hosts.mpi 文件中设置 server ip 地址（注意末尾也要有换行）

```c
10.176.24.160 slots=1
10.176.24.162 slots=1

```

在 global.hpp 中设置存储层相关参数，包括最大 server 数量和最大 thread 数量

```c
int Global::num_servers = 2;
int Global::num_threads = 16;

int Global::rdma_buf_size_mb = 64;
int Global::memstore_size_gb = 20;
```

配置.json参数文件，并在 test 中加载相应参数

```c
para.LoadConfigFromJSON("./deep10M.json");
```

编译构建

```shell
bash build.sh
```

分布式运行

```shell
bash sync.sh && bash run.sh
```
