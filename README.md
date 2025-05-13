# RED-ANNS

To enable RDMA communication, we referred to the RDMA communication code implementation in the [Wukong Project](https://github.com/SJTU-IPADS/wukong) and ported it to the RoCE network.

### Environment Configuration

Install Dependencies

```shell
sudo apt install numactl
sudo apt-get install libboost-all-dev
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

download boost_1_85_0.tar.gz https://www.boost.org/users/history/version_1_85_0.html

```shell
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
tar -xzvf boost_1_85_0.tar.gz
cd boost_1_85_0
./bootstrap.sh --prefix=/usr/
./bootstrap.sh --prefix=/usr/local/
sudo ./b2 install
```

Configure passwordless SSH access from the machine running the current shell to other hosts (including to itself), to facilitate using rsync.

Remove the maximum locked memory limit on all machines.

Configure the MTU of the network interface

### Project Configuration

Set the server IP address in the hosts file (make sure there is also a newline at the end).

```c
10.176.24.160
10.176.24.162

```

Set the server IP address in the hosts.mpi file (ensure there is a newline at the end of the file).

```c
10.176.24.160 slots=1
10.176.24.162 slots=1

```

Set the storage layer-related parameters in global.hpp, including the maximum number of servers and the maximum number of threads.

```c
int Global::num_servers = 2;
int Global::num_threads = 16;

int Global::rdma_buf_size_mb = 64;
int Global::memstore_size_gb = 20;
```

Configure the .json parameter file and load the corresponding parameters in the test.

```c
para.LoadConfigFromJSON("./deep10M.json");
```

Compile and Build

```shell
bash build.sh
```

Run

```shell
bash sync.sh && bash run.sh
```
