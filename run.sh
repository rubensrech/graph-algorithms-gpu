#!/bin/bash
PROG=$1
INPUT=$2
N=$3

ACC=0.0
for ((i=0;i<N;i++)); do
    time=$(./$PROG $INPUT)
    echo $time
    ACC=$(echo "$ACC + $time" | bc)
done
AVG=$(echo "scale=6; $ACC / $N" | bc)
echo "Average time: 0${AVG}"
