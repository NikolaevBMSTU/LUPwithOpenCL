#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



__kernel void lu_kernel(const unsigned int k, const unsigned int N, __global double* A, global int* ip, global int* ier, global int* id, global int* k_act) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	
	const uint j = get_global_id(0);

	// for (uint k = 0; k < N - 1; ++k) { // проход по столбцам

		// одни поток проходит по строкам в столбце и ищет максимальный элемент
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

			k_act[k] = k;

			for (uint i = k + 1; i < N; ++i)
				A[i * N + k] /= -t;
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		// for (int j = k + 1; j < n; ++j) { // этот цикл заменяем на 
		if (j >= k + 1 && j < N) {

			id[j] = j;

			//чушь отладочная
			// for (int i = k + 1; i < N; ++i)
			// 	A[i * N + j] = 1;

			// std::swap(A[m * N + j], A[k * N + j]);
			double tmp = A[ip[k] * N + j];
			A[ip[k] * N + j] = A[k * N + j];
			A[k * N + j] = tmp;

			// barrier(CLK_GLOBAL_MEM_FENCE);

			if (tmp != 0.0)
				for (int i = k + 1; i < N; ++i)
					A[i * N + j] += A[i * N + k] * tmp;
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	// }

	// обратная подстановка

	// for (int k = 0; k < n - 1; ++k) {
	// 	m = ip[k];
	// 	t = b[m];
	// 	b[m] = b[k];
	// 	b[k] = t;
	// 	for (int i = k + 1; i < n; ++i)
	// 		b[i] += A[i][k]*t;
	// }

	// for (int k = 0; k < n - 1; ++k) {
	// 	km1 = n - k - 2;
	// 	kb = km1 + 1;
	// 	b[kb] = b[kb]/A[kb][kb];
	// 	t = -b[kb];
	// 	for (int i = 0; i <= km1; ++i) 
	// 		b[i] += A[i][kb]*t;
	// }
	
	// b[0] = b[0]/A[0][0];

}



);} // ############################################################### end of OpenCL C code #####################################################################

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