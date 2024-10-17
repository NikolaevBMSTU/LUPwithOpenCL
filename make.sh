#!/usr/bin/env bash

mkdir -p bin # create directory for executable
rm -f bin/OpenCL-Wrapper # prevent execution of old version if compiling fails

SRC="src/*.cpp src/Origin/*.cpp src/OpenCL-Wrapper/*.cpp"
INC="-I./src/OpenCL/include -I./src/OpenCL-Wrapper"
OLEVEL=3

case "$(uname -a)" in # automatically detect operating system
	 Darwin*) g++ ${SRC} -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment ${INC} -L./src/OpenCL/lib -lOpenCL     ;; # macOS
	*Android) g++ ${SRC} -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment ${INC} -framework OpenCL               ;; # Android
	*       ) g++ ${SRC} -o bin/OpenCL-Wrapper -std=c++17 -pthread -g -O${OLEVEL} -DOPTIMIZATION_LEVEL=${OLEVEL} -Wno-comment ${INC} -L/system/vendor/lib64 -lOpenCL ;; # Linux
esac

if [[ $? == 0 ]]; then bin/OpenCL-Wrapper "$@"; fi # run executable only if last compilation was successful
