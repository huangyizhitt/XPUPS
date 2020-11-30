#!/bin/bash
# set -x
if [ $# -lt 1 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi
export DMLC_PS_VAN_TYPE='zmq_ipc'
export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=1
bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} ${arg} &


export EPOCH='20'
# start servers
export DMLC_ROLE='server'
i=0
export HEAPPROFILE=./S${i}
export SERVER_XPU_NAME='W-2155'
export SERVER_XPU_TYPE='CPU'
export SERVER_XPU_MAX_CORE='20'
export SERVER_XPU_THREADS='20'
export NUMA_NODE=0
${bin} ${arg} &

# start workers
i=1
export DMLC_ROLE='worker'
export HEAPPROFILE=./W${i}
export WORKER_XPU_NAME='W-2155'
export WORKER_XPU_TYPE='CPU'
export WORKER_XPU_MAX_CORE='40'
export WORKER_XPU_THREADS='18'
export NUMA_NODE=${i}
${bin} ${arg} &

wait