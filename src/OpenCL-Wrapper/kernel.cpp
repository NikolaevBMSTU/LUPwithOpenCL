#include "kernel.hpp" 

string get_opencl_c_code(std::string r) {
	r = replace(r, " ", "\n"); // replace all spaces by new lines
	r = replace(r, "#ifdef\n", "#ifdef "); // except for the arguments after some preprocessor options that need to be in the same line
	r = replace(r, "#ifndef\n", "#ifndef ");
	r = replace(r, "#define\n", "#define "); // #define with two arguments will not work
	r = replace(r, "#if\n", "#if "); // don't leave any spaces in arguments
	r = replace(r, "#elif\n", "#elif "); // don't leave any spaces in arguments
	r = replace(r, "#pragma\n", "#pragma ");
	return "\n"+r;
}
