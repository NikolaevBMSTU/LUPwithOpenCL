#!/bin/bash

BUILD_DIR="bin"

TARGET="benchmark"

mkdir -p ${BUILD_DIR} # create directory for executable

SRC="src/*.cpp src/decsol/*.cpp src/decsol_origin/*.cpp src/decsol_OpenCL/*.cpp src/OpenCL-Wrapper/*.cpp"
INC="-I./external/OpenCL/include -I./external/ViennaCL -I./src/OpenCL-Wrapper"
LIBPATH="-L./external/OpenCL/lib -L./external/OpenBLAS"
LIB="-lOpenCL -pthread -llapacke -llapack -lopenblas -lgfortran -lquadmath"

#Compiler and Linker
if [[ $(uname -a) =~ Cygwin ]]; then
	echo Building for Cygwin...
	CXX=x86_64-w64-mingw32-g++
	LIB+=" -static"
	INC+=" -IF:\\cygwin64\\home\\Vitaly\\.local\\lapack\\include"
	LIBPATH+=" -LF:\\cygwin64\\home\\Vitaly\\.local\\lapack\\lib"
elif [[ $(uname -a) =~ Linux ]]; then
	echo Building for Linux...
	CXX=g++
else
	echo Unknown system
	exit 1
fi


OLEVEL=3
# DEBUG="-g"
# -fopenmp 
DIRECTIVES="-DOPTIMIZATION_LEVEL=${OLEVEL} -DVIENNACL_WITH_OPENCL"
WARNINGS="-Wno-comment -Wno-ignored-attributes"


if [ $# -eq 0 ]; then
    MATRIX_SIZE=3600
else
	MATRIX_SIZE=$1
fi

# Compilation
$CXX ${SRC} -o ${BUILD_DIR}/${TARGET} -std=c++17 ${DEBUG} -O${OLEVEL} ${DIRECTIVES} ${WARNINGS} ${INC} ${LIBPATH} ${LIB}


# Test run
if [[ $? == 0 ]]; then echo "Run..."; ${BUILD_DIR}/${TARGET} ${MATRIX_SIZE}; fi # run executable only if last compilation was successful
