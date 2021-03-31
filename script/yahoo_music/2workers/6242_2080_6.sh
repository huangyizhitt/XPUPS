#!/bin/bash
# set -x
if [ $# -lt 1 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi
export DMLC_PS_VAN_TYPE='zmq_ipc'
export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=2
bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} ${arg} &


export EPOCH='20'
export TRANSMODE=6
export DATA_PATH="yahoo_music.bin"
# start servers
export DMLC_ROLE='server'
i=0
export HEAPPROFILE=./S${i}
export XPU_NAME='Gold 6242'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=32
export XPU_WORKERS=2
export NUMA_NODE=0
${bin} ${arg} &

# start workers
i=1
export DMLC_ROLE='worker'
export HEAPPROFILE=./W${i}
export XPU_NAME='Gold 6242'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=32
export XPU_WORKERS=16
export NUMA_NODE=${i}
export WORK_LOAD=18
${bin} ${arg} &

export DMLC_ROLE='worker'
export HEAPPROFILE=./W${i}
export XPU_NAME='GeForce RTX 2080'
export XPU_TYPE='GPU'
export XPU_MAX_CORES=2944
export XPU_WORKERS=2576
export NUMA_NODE=0
export DEVICE_ID=1
export WORK_LOAD=82
${bin} ${arg} &

wait
