#include "OpenCL-Wrapper/opencl.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

#include "Origin/decsol.hpp"

class Timer {

	public:
		Timer() : ts {} {}

		void start() {
			ts = std::chrono::system_clock::now() ;
		}

		double get() const {
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - ts;
			return elapsed_seconds.count();
		}

	private:
		std::chrono::_V2::system_clock::time_point ts;
};

// Matrix Triangularization by Gaussian Elimination
template <typename T>
void dec_optimized(const int n, T **A, int *ip, int* ier)
{

/*-----------------------------------------------------------------------
	Matrix Triangularization by Gaussian Elimination

	Input:
		n		order of matrix
		A		matrix to be triangularized

	Output:
		A[i][j], i <= j		upper triangular factor, U
		A[i][j], i > j		multipliers = lower triangular factor, i - l
		ip[k], k < n - 1	index of k-th pivot rowf
		ip[n-1]				(-1)^(number of interchanges) or 0
		ier 				0 if matrix A is nonsingular, or k if found
								to be singular at stage k

	Use sol to obtain solution of linear system

	determ(A) = ip[n-1]*A[0][0]*A[1][1]*...*A[n-1][n-1]

	If ip[n-1] = 0, A is singular, sol will divide by zero

	Reference:
		C.B. Moler, Algorithm 423, Linear Equation Solver,
		C.A.C.M. 15 (1972), p. 274.
-----------------------------------------------------------------------*/

	int m, k;
	T t;

	*ier = 0;
	ip[n-1] = 1; // детерминант
	if (n != 1) {

		// kernel starts
		for (k = 0; k < n - 1; ++k) { // проход по столбцам

			for (int j = k + 1; j < n; ++j) { // этот цикл заменяем на 

				// одни поток проходит по строкам в столбце и ищет максимальный элемент
				if (j == k + 1) {
					m = k;
					for (int i = k + 1; i < n; ++i) {
						if (fabs(A[i][k]) > fabs(A[m][k])) m = i;
					}
					ip[k] = m;
					t = A[m][k];
					if (m != k) {
						ip[n-1] = -ip[n-1]; // детерминант
						A[m][k] = A[k][k];
						A[k][k] = t;
					}
					if (t == 0.0) {
						*ier = k;
						ip[n-1] = 0;
						return;
					}
					// t = 1.0/t;

					for (int i = k + 1; i < n; ++i)
						A[i][k] /= -t;
				}

				// синхронизация

				std::swap(A[m][j], A[k][j]);

				if (t != 0.0)
					for (int i = k + 1; i < n; ++i)
						A[i][j] += A[i][k]*A[k][j];
			}
		}

		// конец kernel
	}

	if (A[n-1][n-1] == 0.0) {
		*ier = n;
		ip[n-1] = 0;
	}

	return;

} //dec


template <typename T>
void sol_optimized(const int n, T **A, T *b, int *ip)
{

/*-----------------------------------------------------------------------
	Solution of linear system A*x = b

	Input:
		n		order of matrix
		A		triangularized matrix obtained from dec
		b		right hand side vector
		ip		pivot vector obtained from dec

	Do not use if dec has set ier != 0

	Output:
		b		solution vector, x

-----------------------------------------------------------------------*/

	int m, kb, km1;
	T t;

	if (n != 1) {
		for (int k = 0; k < n - 1; ++k) {
			m = ip[k];
			t = b[m];
			b[m] = b[k];
			b[k] = t;
			for (int i = k + 1; i < n; ++i)
				b[i] += A[i][k]*t;
		}
		for (int k = 0; k < n - 1; ++k) {
			km1 = n - k - 2;
			kb = km1 + 1;
			b[kb] = b[kb]/A[kb][kb];
			t = -b[kb];
			for (int i = 0; i <= km1; ++i) 
				b[i] += A[i][kb]*t;
		}
	}
	b[0] = b[0]/A[0][0];
	
	return;
	
} //sol

template<typename T>
void print_vector(const uint N, T* A) {
	for (uint i = 0; i < N; i++) {
		std::cout << A[i] << " ";
	}
	std::cout << "\n";
}

template <typename T>
void print_matrix(const uint N, const uint M, T* A) {
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < M; j++)
			std::cout << A[i*N+j] << " ";
		std::cout << "\n";
	}
}

template<typename T>
void print_matrix(const uint N, const uint M, T** A) {
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < M; j++)
			std::cout << A[i][j] << " ";
		std::cout << "\n";
	}
}

template<typename T>
void print_matrix_along(const uint N, const uint M, T** A, T** B) {
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < M; j++)
			std::cout << A[i][j] << " ";
		std::cout << "\t\t";
		for (uint j = 0; j < M; j++)
			std::cout << B[i][j] << " ";
		std::cout << "\n";
	}
}

int main() {

	Timer timer {};
	const uint N = 576u; // size of vectors
	#undef ORIGIN_TEST
	#define BENCHMARK 1
	#define COMPARE 1

	// =====================================
	// ==== Решение в исходном варианте ====
	// =====================================

#ifdef ORIGIN_TEST
	double** CPU_A = new double*[N];
	double** CPU_B = new double*[N];
	int	CPU_ip[N];
	int CPU_ier = 0;

	for(uint i = 0; i < N; i++) {
		CPU_A[i] = new double[N];
		for (uint j = 0; j < N; j++)
			CPU_A[i][j] = (i * N + j + 1) * 1.0f;
	}

	// Для повышения устойчивости?
	std::swap(CPU_A[N-1][N-2], CPU_A[N-1][N-1]);

	for(uint i = 0; i < N; i++) {
		CPU_B[i] = new double[N];
		for (uint j = 0; j < N; j++)
			CPU_B[i][j] = CPU_A[i][j];
	}

	for(uint n=0u; n<N; n++) {
		CPU_ip[n] = 0; // initialize memory
	}

	print_matrix_along(N, N, CPU_A, CPU_B);
	std::cout << "\n";

	CPU_ier = dec(N, CPU_A, CPU_ip);
	dec_optimized(N, CPU_B, CPU_ip, &CPU_ier);

	if (CPU_ier != 0) std::cout << "Ошибка в результате разложения" ;

	print_matrix_along(N, N, CPU_A, CPU_B);

	print_vector(N, CPU_ip);
#endif

#ifdef BENCHMARK
	double** CPU_ORIGIN = new double*[N];
	double** CPU_A = new double*[N];
	int	CPU_ip[N];
	int CPU_ier = 0;

	for(uint i = 0; i < N; i++) {
		CPU_ORIGIN[i] = new double[N];
		for (uint j = 0; j < N; j++)
			CPU_ORIGIN[i][j] = drand48() * 1000;
	}

	for(uint i = 0; i < N; i++) {
		CPU_A[i] = new double[N];
		for (uint j = 0; j < N; j++)
			CPU_A[i][j] = CPU_ORIGIN[i][j];
	}

	for(uint n=0u; n<N; n++) {
		CPU_ip[n] = 0; // initialize memory
	}

	timer.start();
	CPU_ier = dec(N, CPU_A, CPU_ip);
	auto elapsed_time = timer.get();

	std::cout << "Spend time = " << elapsed_time << std::endl;
	// print_vector(N, CPU_ip);

#endif

	std::cout << "============" << std::endl;
	std::cout << "== OpenCL ==" << std::endl;
	std::cout << "============" << std::endl;


	// ========================
	// ======== OpenCL ========
	// ========================

	// for (auto device : get_devices()) {
	// 	std::cout << device.name << "\n";
	// }


	Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

	Memory<double> GPU_A(device, N * N); // allocate memory on both host and device
	Memory<int>	  GPU_ip(device, N);
	Memory<int>   ier(device, 1);

	Memory<int>   id(device, N);
	Memory<int>   k_act(device, N);

	// Kernel lu_kernel(device, N, "lu_kernel", N, A, ip, ier); // kernel that runs on the device

#ifdef ORIGIN_TEST
	for(uint n=0u; n<N * N; n++) {
		A[n] = (n + 1) * 1.0f; // initialize memory
	}

	// Для повышения устойчивости?
	std::swap(A[N * N - 2], A[N * N - 1]);
#endif

#ifdef BENCHMARK
	
	for (uint i=0u; i<N; i++) {
		for(uint j=0u; j<N; j++) {
			GPU_A[i * N + j] = CPU_ORIGIN[i][j]; // initialize memory
		}
	}

#endif

	for(uint n=0u; n<N; n++) {
		GPU_ip[n] = 1; // initialize memory
	}

	timer.start();

	GPU_A.write_to_device();
	GPU_ip.write_to_device();
	ier.write_to_device();
	
	for (uint k = 0; k < N-1; k++) {
		Kernel lu_kernel(device, N, N, "lu_kernel", k, N, GPU_A, GPU_ip, ier, id, k_act);
		lu_kernel.enqueue_run();
	}

	device.finish_queue();

	GPU_A.read_from_device(); // copy data from device memory to host memory
	GPU_ip.read_from_device();
	ier.read_from_device();
	id.read_from_device();
	k_act.read_from_device();

	elapsed_time = timer.get();

	std::cout << "Spend time = " << elapsed_time << std::endl;
	std::cout << "Result ier = " << ier[0] << std::endl;

	// std::cout << "id work-items\n";
	// print_vector(N, id.data());
	// std::cout << "k\n";
	// print_vector(N, k_act.data());

	std::cout << std::endl;

	// std::cout << "Сравнение CPU_ip и GPU_ip\n";
	// print_vector(N, CPU_ip);
	// print_vector(N, GPU_ip.data());

	// print_matrix(3, N, &GPU_A[0]);

#ifdef COMPARE

	for (uint i=0u; i<N; i++) {
		if (CPU_ip[i] != GPU_ip[i])
			throw std::runtime_error("Wrong result IP at i = " + std::to_string(i) +
				" Expected: " + std::to_string(CPU_ip[i]) + " but actual is " + std::to_string(GPU_ip[i]));
	}

	for (uint i=0u; i<N; i++) {
		for(uint j=0; j<N; j++) { 
			if (std::abs((GPU_A[i * N + j] - CPU_A[i][j]) / GPU_A[i * N + j]) > 1e-6)
				throw std::runtime_error("Wrong result LU at i = " + std::to_string(i) + " j = " + std::to_string(j) +
					" Expected: " + std::to_string(GPU_A[i * N + j]) + " but actual is " + std::to_string(CPU_A[i][j]));
		}
	}

#endif

#ifdef ORIGIN_TEST
	print_matrix(N, N, &A[0]);
	std::cout << std::endl;
	
	print_vector(N, &ip[0]);
	std::cout << "ier = " << ier[0] << std::endl;
#endif

#ifdef BENCHMARK

#endif

	// print_info("Value after kernel execution: C[0] = "+to_string(C[0]));

	// wait();
	return 0;
}
