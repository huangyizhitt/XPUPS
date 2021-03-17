#! /bin/bash

log_dir="log"
test -d ${log_dir} || mkdir -p ${log_dir}
test_times=100
file=${log_dir}/$(date "+%Y%m%d%H%M%S")-netflix

for i in $(seq 1 ${test_times})
do
	./script/netflix/1worker/6242_6.sh ./mf ${file}-6242.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/netflix/1worker/6242l_6.sh ./mf ${file}-6242-local.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/netflix/1worker/2080_6.sh ./mf ${file}-2080.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/netflix/1worker/2080s_6.sh ./mf ${file}-2080s.csv
done

