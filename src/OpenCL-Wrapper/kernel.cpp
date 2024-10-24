#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
#include "syntax_highlight.hpp"

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

string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void lu_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint j = get_global_id(0);

	if (j >= k + 1) { // позволяет использовать working_groups, а общее work_items может быть больше N

		// std::swap(A[m * N + j], A[k * N + j]);
		double tmp = A[ip[k] * N + j];
		A[ip[k] * N + j] = A[k * N + j];
		A[k * N + j] = tmp;

		if (tmp != 0.0)
			for (int i = k + 1; i < N; ++i)
				A[i * N + j] += A[i * N + k] * tmp;
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################


// С этим ядром работает медленнее. Видимо деплой занимает много времени
string opencl_c_container_2() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void lu_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	// offset = k + 1, work_items_number = N - k - 1 

	const uint j = get_global_id(0);

	double tmp = A[ip[k] * N + j];
	A[ip[k] * N + j] = A[k * N + j];
	A[k * N + j] = tmp;

	if (tmp != 0.0)
		for (int i = k + 1; i < N; ++i)
			A[i * N + j] += A[i * N + k] * tmp;

}


);} // ############################################################### end of OpenCL C code #####################################################################



string opencl_c_container_pivot_kernel() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void pivot_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint j = get_global_id(0);

	if (j == 0) {
		int m = k;
		for (int i = k + 1; i < N; ++i) {
			if (fabs(A[i * N + k]) > fabs(A[m * N + k])) m = i;
		}
		ip[k] = m;
		double t = A[m * N + k];
		if (m != k) {
			ip[N-1] = -ip[N-1]; // детерминант
			A[m * N + k] = A[k * N + k];
			A[k * N + k] = t;
		}
		if (t == 0.0) {
			*ier = k;
			ip[N-1] = 0;
			return;
		}

		for (uint i = k + 1; i < N; ++i)
			A[i * N + k] /= -t;
	}
}


);} // ############################################################### end of OpenCL C code #####################################################################


string find_max_in_column_kernel() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void find_max_in_column_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip,
		global int* ier, __local int* tmp_ips) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint lid = get_global_id(0);

	bool check_last = false;

	int group_size = N - (k + 1) + 1;

	if (group_size % 2 != 0) {
		check_last = true;
		group_size--;
	}

	tmp_ips[lid] = k + lid;

	for (size_t i = group_size / 2; i > 0; i >>= 1) {
		if (lid < i) {
			if (fabs(A[tmp_ips[lid] * N + k]) > fabs(A[(k + lid + i) * N + k])) {
				tmp_ips[lid] = k + lid;
			} else {
				tmp_ips[lid] = k + lid + i;
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid == 0) {

		if (check_last) {
			if (fabs(A[(N - 1) * N + k]) > fabs(A[tmp_ips[lid] * N + k])) {
				tmp_ips[lid] = N - 1;
			}
		}

		int m = tmp_ips[lid];

		if (m != k) {
			ip[N-1] = -ip[N-1]; // детерминант

			double t = A[m * N + k];
			A[m * N + k] = A[k * N + k];
			A[k * N + k] = t;
		}

		ip[k] = m ; // save row number of max element

		// swap rows if needed
		if (m != k) {
			ip[N-1] = -ip[N-1]; // детерминант?

			//
			double t = A[m * N + k];
			A[m * N + k] = A[k * N + k];
			A[k * N + k] = t;

			if (t == 0.0) {
				*ier = k;
				ip[N-1] = 0;
				return;
			}

			// todo add row swap for vector b
		}
	}
}


);} // ############################################################### end of OpenCL C code #####################################################################


string find_max_in_column_kernel_2() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void find_max_in_column_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	int m = k;
	for (int i = k + 1; i < N; ++i) {
		if (fabs(A[i * N + k]) > fabs(A[m * N + k])) m = i;
	}

	ip[k] = m ; // save row number of max element

	// swap rows if needed
	if (m != k) {
		ip[N-1] = -ip[N-1]; // детерминант?

		// swap diagonal element A
		double t = A[m * N + k];
		A[m * N + k] = A[k * N + k];
		A[k * N + k] = t;

		if (t == 0.0) {
			*ier = k;
			ip[N-1] = 0;
			return;
		}
	}
}


);} // ############################################################### end of OpenCL C code #####################################################################


string identify_column_kernel() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void identify_column_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	// for (uint i = k + 1; i < N; ++i)
	if (i >= k + 1 && i < N)
		A[i * N + k] /= - A[k * N + k];

}


);} // ############################################################### end of OpenCL C code #####################################################################



string swap_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void swap_kernel(const unsigned int k, __global double* b, __global int* ip) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	int m = ip[k];
	double t = b[m];
	b[m] = b[k];
	b[k] = t;

}


);} // ############################################################### end of OpenCL C code #####################################################################





string forward_substitution() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void forward_substitution_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	if (i >= k + 1 && i < N) {
		b[i] += A[i * N + k] * b[k];
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################


string devide_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void devide_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	uint kb = N - k - 1;
	b[kb] /= A[kb * N + kb];

}


);} // ############################################################### end of OpenCL C code #####################################################################


string back_substitution() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void back_substitution_kernel(const unsigned int k, const unsigned int N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	uint kb = N - k - 1;

	if (i >= 0 && i < kb) {
		b[i] += A[i * N + kb] * -b[kb];
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################
