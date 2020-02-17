#!/bin/bash
for i in {6..22}
do
    for j in {1..13}
    do
        echo $i $j
        nohup python3 correlation_learning.py $i $j > ./output_{$i}_{$j}.txt 2>&1 &
    done
done

# ps -ef | grep num_var | grep -v grep | cut -c 9-15 | xargs kill -s 9