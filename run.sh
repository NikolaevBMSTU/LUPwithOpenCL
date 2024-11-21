#!/bin/bash

TARGET="bin/benchmark"

COUNT=3

for SIZE in 576
do
	for j in 1 .. ${COUNT}
	do
		echo 
		${TARGET} ${SIZE}
	done
done
