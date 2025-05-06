if [ -d "./build" ]; then
    echo "./build directory already existed"
else
    mkdir ./build
fi

cd ./build
cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release .. # Release 版本记得打开 O3 优化
make -j
cd ..
