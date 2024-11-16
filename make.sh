#!/usr/bin/env bash

TARGET="bin/OpenCL-Wrapper"

mkdir -p bin # create directory for executable
rm -f ${TARGET} # prevent execution of old version if compiling fails

SRC="src/*.cpp src/Origin/*.cpp src/Origin_New/*.cpp src/Own/*.cpp src/OpenCL-Wrapper/*.cpp"
INC="-I./src/OpenCL/include -I./src/OpenCL-Wrapper -IViennaCL-1.7.1"
OLEVEL=3
DEBUG="-g"

if [ $# -eq 0 ]; then
    MATRIX_SIZE=576
else
	MATRIX_SIZE=$1
fi

case "$(uname -a)" in # automatically detect operating system
	 Darwin*) g++ ${SRC} -o ${TARGET} -std=c++17 -pthread ${DEBUG} -DMATRIX_SIZE=${MATRIX_SIZE} -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL -Wno-comment ${INC} -L./src/OpenCL/lib -lOpenCL     ;; # macOS
	*Android) g++ ${SRC} -o ${TARGET} -std=c++17 -pthread ${DEBUG} -DMATRIX_SIZE=${MATRIX_SIZE} -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL -Wno-comment ${INC} -framework OpenCL               ;; # Android
	*       ) g++ ${SRC} -o ${TARGET} -std=c++17 -pthread ${DEBUG} -DMATRIX_SIZE=${MATRIX_SIZE} -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL -Wno-comment ${INC} -L/system/vendor/lib64 -lOpenCL ;; # Linux
esac

if [[ $? == 0 ]]; then ${TARGET} ${MATRIX_SIZE} "$@"; fi # run executable only if last compilation was successful
