
#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "../OpenCL-Wrapper/utilities.hpp"
#define R(...) string(" "#__VA_ARGS__" ") // evil stringification macro, similar syntax to raw string R"(...)"

string column_opencl_c_container(); // outsourced to kernel.cpp
string column_find_max_in_column_kernel_2(); // outsourced to kernel.cpp
string column_identify_column_kernel(); // outsourced to kernel.cpp
string column_swap_kernel_code(); // outsourced to kernel.cpp
string column_devide_kernel_code(); // outsourced to kernel.cpp
string column_forward_substitution(); // outsourced to kernel.cpp
string column_back_substitution(); // outsourced to kernel.cpp

// string get_opencl_c_code(std::string r);

#endif
