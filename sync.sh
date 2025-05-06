#!/bin/bash
#e.g. ./sync.sh

# 配置你的 sync 的目录
export WUKONG_ROOT=/ann/PQ/RED-ANNS
root=${WUKONG_ROOT}/
if [ "$root" = "/" ]; then
    echo -e "\e[31m PLEASE set WUKONG_ROOT! \e[0m"
    exit 0
fi

cat hosts | while read machine; do
    echo "sync file on" ${machine}
    rsync -rtuvl --exclude=.git ${root} ${machine}:${root}
done
