#pragma once

#include "utilities.hpp"
#define R(...) string(" "#__VA_ARGS__" ") // evil stringification macro, similar syntax to raw string R"(...)"

string get_opencl_c_code(std::string r);
