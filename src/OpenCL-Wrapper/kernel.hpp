#pragma once

#include "utilities.hpp"
#define R(...) string(" "#__VA_ARGS__" ") // evil stringification macro, similar syntax to raw string R"(...)"

string opencl_c_container(); // outsourced to kernel.cpp
string opencl_c_container_2(); // outsourced to kernel.cpp
string opencl_c_container_pivot_kernel(); // outsourced to kernel.cpp
string find_max_in_column_kernel(); // outsourced to kernel.cpp
string find_max_in_column_kernel_2(); // outsourced to kernel.cpp
string identify_column_kernel(); // outsourced to kernel.cpp
string swap_kernel_code(); // outsourced to kernel.cpp
string devide_kernel_code(); // outsourced to kernel.cpp
string forward_substitution(); // outsourced to kernel.cpp
string back_substitution(); // outsourced to kernel.cpp

string get_opencl_c_code(std::string r);
