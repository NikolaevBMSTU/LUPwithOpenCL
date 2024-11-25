#!/bin/bash

#Compiler and Linker
if [[ $(uname -a) =~ Cygwin ]]; then
	echo Building for Cygwin...
	CXX=x86_64-w64-mingw32-g++
	LIB+=" -static"
elif [[ $(uname -a) =~ Linux ]]; then
	echo Building for Linux...
	CXX=g++
else
	echo Unknown system
	exit 1
fi

#Compile
$CXX plot/plot_prepare.cpp -o bin/plot_prepare -Isrc -static

#Prepare
bin/plot_prepare

#Plot
gnuplot plot/plot.gp
