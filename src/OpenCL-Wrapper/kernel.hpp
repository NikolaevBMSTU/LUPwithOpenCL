#pragma once

#include "utilities.hpp"
#define R(...) string(" "#__VA_ARGS__" ") // evil stringification macro, similar syntax to raw string R"(...)"

string opencl_c_container(); // outsourced to kernel.cpp
string opencl_c_container_pivot_kernel(); // outsourced to kernel.cpp

string get_opencl_c_code(std::string r);
