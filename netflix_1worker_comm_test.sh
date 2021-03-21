#! /bin/bash

log_dir="log"
test -d ${log_dir} || mkdir -p ${log_dir}
test_times=20
file=${log_dir}/$(date "+%Y%m%d%H%M%S")-netflix

for i in $(seq 0 6)
do
	for j in $(seq 1 ${test_times})
	do
		./script/netflix/1worker/6242_${i}.sh ./mf ${file}-comm-6242-${i}.csv
	done
done

for i in $(seq 0 6)
do
        for j in $(seq 1 ${test_times})
        do
                ./script/netflix/1worker/2080s_${i}.sh ./mf ${file}-comm-2080s-${i}.csv
        done
done
