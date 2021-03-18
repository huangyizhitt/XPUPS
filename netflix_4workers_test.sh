#! /bin/bash

log_dir="log"
test -d ${log_dir} || mkdir -p ${log_dir}
test_times=100
file=${log_dir}/$(date "+%Y%m%d%H%M%S")-netflix

for i in $(seq 1 ${test_times})
do
	./script/netflix/4workers/6242_6242l_2080_2080s_3.sh ./mf ${file}-6242_6242l_2080_2080s_3.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/netflix/4workers/6242_6242l_2080_2080s_4.sh ./mf ${file}-6242_6242l_2080_2080s_4.csv
done

for i in $(seq 1 ${test_times})
do
        ./script/netflix/4workers/6242_6242l_2080_2080s_6.sh ./mf ${file}-6242_6242l_2080_2080s_6.csv
done


