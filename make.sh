#!/usr/bin/env bash

TARGET="bin/OpenCL-Wrapper"

mkdir -p bin # create directory for executable
rm -f ${TARGET} # prevent execution of old version if compiling fails

SRC="src/*.cpp src/Origin/*.cpp src/OpenCL-Wrapper/*.cpp"
INC="-I./src/OpenCL/include -I./src/OpenCL-Wrapper -IViennaCL-1.7.1"
OLEVEL=3
DEBUG="-g"

case "$(uname -a)" in # automatically detect operating system
	 Darwin*) g++ ${SRC} -o ${TARGET} -std=c++17 -pthread ${DEBUG} -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL -Wno-comment ${INC} -L./src/OpenCL/lib -lOpenCL     ;; # macOS
	*Android) g++ ${SRC} -o ${TARGET} -std=c++17 -pthread ${DEBUG} -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL -Wno-comment ${INC} -framework OpenCL               ;; # Android
	*       ) g++ ${SRC} -o ${TARGET} -std=c++17 -pthread ${DEBUG} -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL -Wno-comment ${INC} -L/system/vendor/lib64 -lOpenCL ;; # Linux
esac

if [[ $? == 0 ]]; then ${TARGET} "$@"; fi # run executable only if last compilation was successful
