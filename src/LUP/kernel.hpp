
#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "../OpenCL-Wrapper/utilities.hpp"
#define R(...) string(" "#__VA_ARGS__" ") // evil stringification macro, similar syntax to raw string R"(...)"

string lu_kernel_code();
string find_max_in_column_kernel();
string identify_column_kernel();
string swap_kernel_code();
string devide_kernel_code();

string forward_substitution();
string back_substitution();

#endif
