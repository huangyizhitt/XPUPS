#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi
export DMLC_PS_VAN_TYPE='shm'
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

# start the scheduler
rank=0
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=${rank}
export DMLC_ROLE='scheduler'
export DMLC_SHM_ID=${rank}
${bin} ${arg} &
rank=`expr ${rank} + 1`

export EPOCH='20'
# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    export HEAPPROFILE=./S${i}
    export SERVER_XPU_NAME='W-2155'
    export SERVER_XPU_TYPE='CPU'
    export SERVER_XPU_MAX_CORE='20'
    export SERVER_XPU_THREADS='20'
    export DMLC_SHM_ID=${rank}
    export NUMA_NODE=${i}
    rank=`expr ${rank} + 1`
    ${bin} ${arg} &
done

# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    export WORKER_XPU_NAME='W-2155'
    export WORKER_XPU_TYPE='CPU'
    export WORKER_XPU_MAX_CORE='20'
    export WORKER_XPU_THREADS='2'
    export DMLC_SHM_ID=${rank}
    export NUMA_NODE=${i}
    rank=`expr ${rank} + 1`
    ${bin} ${arg} &
done

wait
