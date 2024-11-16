#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
#include "../OpenCL-Wrapper/syntax_highlight.hpp"

// string get_opencl_c_code(std::string r) {
// 	r = replace(r, " ", "\n"); // replace all spaces by new lines
// 	r = replace(r, "#ifdef\n", "#ifdef "); // except for the arguments after some preprocessor options that need to be in the same line
// 	r = replace(r, "#ifndef\n", "#ifndef ");
// 	r = replace(r, "#define\n", "#define "); // #define with two arguments will not work
// 	r = replace(r, "#if\n", "#if "); // don't leave any spaces in arguments
// 	r = replace(r, "#elif\n", "#elif "); // don't leave any spaces in arguments
// 	r = replace(r, "#pragma\n", "#pragma ");
// 	return "\n"+r;
// }

 string column_opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void lu_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint j = get_global_id(0);

	if (j >= k + 1 && j < N) { // позволяет использовать working_groups, а общее work_items может быть больше N

		// std::swap(A[m * N + j], A[k * N + j]);

		__private int p_ip = ip[k];

		double tmp = A[p_ip + N * j];
		A[p_ip + N * j] = A[k + N * j];
		A[k + N * j] = tmp;

		if (tmp != 0.0)
			for (int i = k + 1; i < N; ++i)
				A[i + N * j] += A[i + N * k] * tmp;
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################


 string column_find_max_in_column_kernel_2() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void find_max_in_column_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	int m = k;
	for (int i = k + 1; i < N; ++i) {
		if (fabs(A[i + N * k]) > fabs(A[m + N * k])) m = i;
	}

	ip[k] = m ; // save row number of max element

	// swap rows if needed
	if (m != k) {
		ip[N-1] = -ip[N-1]; // детерминант?

		// swap diagonal element A
		double t = A[m + N * k];
		A[m + N * k] = A[k + N * k];
		A[k + N * k] = t;

		if (t == 0.0) {
			*ier = k;
			ip[N-1] = 0;
			return;
		}
	}
}


);} // ############################################################### end of OpenCL C code #####################################################################


 string column_identify_column_kernel() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void identify_column_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	if (i >= k + 1 && i < N)
		A[i + N * k] /= - A[k + N * k];

}


);} // ############################################################### end of OpenCL C code #####################################################################



 string column_swap_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void swap_kernel(const unsigned int k, __global double* b, __global int* ip) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	int m = ip[k];
	double t = b[m];
	b[m] = b[k];
	b[k] = t;

}


);} // ############################################################### end of OpenCL C code #####################################################################





 string column_forward_substitution() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void forward_substitution_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	if (i >= k + 1 && i < N) {
		b[i] += A[i + N * k] * b[k];
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################


 string column_devide_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void devide_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	uint kb = N - k - 1;
	b[kb] /= A[kb + N * kb];

}


);} // ############################################################### end of OpenCL C code #####################################################################


 string column_back_substitution() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void back_substitution_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	uint kb = N - k - 1;

	if (i >= 0 && i < kb) {
		b[i] += A[i + N * kb] * -b[kb];
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################

