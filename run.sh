#!/usr/bin/env bash

TARGET="bin/OpenCL-Wrapper"

if [ $# -eq 0 ]; then
    MATRIX_SIZE=576
else
	MATRIX_SIZE=$1
fi

for size in 576
do
	for j in 1 .. 1
	do
		${TARGET} ${size}
	done
done
