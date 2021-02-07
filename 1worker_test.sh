#! /bin/bash

log_dir="log"
test -d ${log_dir} || mkdir -p ${log_dir}
test_times=10
file=${log_dir}/$(date "+%Y%m%d%H%M%S")

for i in $(seq 1 ${test_times})
do
	./script/6242.sh ./mf ${file}-6242.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/6242-local.sh ./mf ${file}-6242-local.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/2080.sh ./mf ${file}-2080.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/2080s.sh ./mf ${file}-2080s.csv
done

