#!/bin/bash
# set -x
if [ $# -lt 1 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi
export DMLC_PS_VAN_TYPE='zmq_ipc'
export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=32
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
# start servers
export DMLC_ROLE='server'
export XPU_NAME='Gold 6242'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=32
export XPU_WORKERS=2
export NUMA_NODE=0
${bin} ${arg} &

# start workers

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &

export DMLC_ROLE='worker'
export XPU_NAME='Gold 6242-1T'
export XPU_TYPE='CPU'
export XPU_MAX_CORES=1
export XPU_WORKERS=1
export NUMA_NODE=1
export WORK_LOAD=1
${bin} ${arg} &
wait
