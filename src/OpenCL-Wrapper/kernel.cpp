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

	if (j >= k + 1 && j < N) {

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


// Новое ядро поиска

// одни поток проходит по строкам в столбце и ищет максимальный элемент
// if (j == 0) {
// 	int m = k;
// 	for (int i = k + 1; i < N; ++i) {
// 		if (fabs(A[i * N + k]) > fabs(A[m * N + k])) m = i;
// 	}
// 	ip[k] = m;
// 	double t = A[m * N + k];
// 	if (m != k) {
// 		ip[N-1] = -ip[N-1]; // детерминант
// 		A[m * N + k] = A[k * N + k];
// 		A[k * N + k] = t;
// 	}
// 	if (t == 0.0) {
// 		*ier = k;
// 		ip[N-1] = 0;
// 		return;
// 	}

// 	k_act[k] = k;

// 	for (uint i = k + 1; i < N; ++i)
// 		A[i * N + k] /= -t;
// }



// РАБОЧИЙ ВАРИАНТ для N < cores_number
// const uint j = get_global_id(0);

// 	for (uint k = 0; k < N - 1; ++k) { // проход по столбцам

// 		// одни поток проходит по строкам в столбце и ищет максимальный элемент
// 		if (j == 0) {
// 			int m = k;
// 			for (int i = k + 1; i < N; ++i) {
// 				if (fabs(A[i * N + k]) > fabs(A[m * N + k])) m = i;
// 			}
// 			ip[k] = m;
// 			double t = A[m * N + k];
// 			if (m != k) {
// 				ip[N-1] = -ip[N-1]; // детерминант
// 				A[m * N + k] = A[k * N + k];
// 				A[k * N + k] = t;
// 			}
// 			if (t == 0.0) {
// 				*ier = k;
// 				ip[N-1] = 0;
// 				return;
// 			}

// 			for (int i = k + 1; i < N; ++i)
// 				A[i * N + k] /= -t;
// 		}

// 		barrier(CLK_GLOBAL_MEM_FENCE);

// 		if (j >= k + 1 && j < N) {

// 			// std::swap(A[m * N + j], A[k * N + j]);
// 			double tmp = A[ip[k] * N + j];
// 			A[ip[k] * N + j] = A[k * N + j];
// 			A[k * N + j] = tmp;

// 			if (tmp != 0.0)
// 				for (int i = k + 1; i < N; ++i)
// 					A[i * N + j] += A[i * N + k] * tmp;
// 		}

// 		barrier(CLK_GLOBAL_MEM_FENCE);