#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(

#include "../OpenCL-Wrapper/syntax_highlight.hpp"


 string lu_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void lu_kernel(const uint k, const uint N, __global double* A, __global int* ip, __global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
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


string find_max_in_column_kernel() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void find_max_in_column_kernel(const uint k, const uint N, __global double* A, __global int* ip, __global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	int m = k;
	for (int i = k + 1; i < N; ++i) {
		if (fabs(A[i + N * k]) > fabs(A[m + N * k])) m = i;
	}

	ip[k] = m ; // save row number of max element

	// swap rows if needed
	if (m != k) {
		ip[N-1] *= -1; // детерминант?

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

string find_max_in_column_kernel_opt() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void find_max_in_column_kernel(const uint k, const uint N, __global double* A, __global int* ip, __global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint lid = get_global_id(0);

	const uint global_size = N - k; // количество элементов, которое надо сравнить

	const uint cores = 128;

	__local int partial[128];

	if (global_size > 2 * cores) {
		__private uint group_size = global_size / cores + 1;

		partial[lid] = lid * group_size;

		for (int i = 1; i < group_size; i++) {
			if (lid * group_size + i < N) {
				partial[lid] = A[partial[lid]] > A[lid * group_size + i] ? partial[lid] : lid * group_size + i;
			}
		}

	} else {
		// первое сравнение, заполняем partial
		__private int i = global_size / 2;
		if (lid + i < N) {
			partial[lid] = A[lid] > A[lid + i] ? lid : lid + i;
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	int start = global_size > 2 * cores ? cores / 2 : global_size / 4;

	// основной цикл сравнения
	for (int i = start; i > 0; i >>= 1) {
		if (lid < i) {
			partial[lid] = A[partial[lid]] > A[partial[lid + i]] ? partial[lid] : partial[lid + i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	// int m = k;
	// for (int i = k + 1; i < N; ++i) {
	// 	if (fabs(A[i + N * k]) > fabs(A[m + N * k])) m = i;
	// }

	if (lid == 0) {
		int m = partial[0];
		ip[k] = m ; // save row number of max element

		// swap rows if needed
		if (m != k) {
			ip[N-1] *= -1; // детерминант?

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
}


);} // ############################################################### end of OpenCL C code #####################################################################



string identify_column_kernel() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void identify_column_kernel(const unsigned int k, const unsigned int N, __global double* A, __global int* ip, __global int* ier) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	if (i >= k + 1 && i < N)
		A[i + N * k] /= - A[k + N * k];

}


);} // ############################################################### end of OpenCL C code #####################################################################



 string swap_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void swap_kernel(const uint k, __global double* b, __global int* ip) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	int m = ip[k];
	double t = b[m];
	b[m] = b[k];
	b[k] = t;

}


);} // ############################################################### end of OpenCL C code #####################################################################





 string forward_substitution() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void forward_substitution_kernel(const uint k, const uint N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	if (i >= k + 1 && i < N) {
		b[i] += A[i + N * k] * b[k];
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################


 string devide_kernel_code() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void devide_kernel(const uint k, const uint N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	uint kb = N - k - 1;
	b[kb] /= A[kb + N * kb];

}


);} // ############################################################### end of OpenCL C code #####################################################################


 string back_substitution() { return R( // ########################## begin of OpenCL C code ####################################################################


__kernel void back_substitution_kernel(const uint k, const uint N, __global double* A, __global double* b) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint i = get_global_id(0);

	const uint kb = N - k - 1;

	// i >= 0 always because i is uint
	if (i < kb) {
		b[i] -= A[i + N * kb] * b[kb];
	}

}


);} // ############################################################### end of OpenCL C code #####################################################################

