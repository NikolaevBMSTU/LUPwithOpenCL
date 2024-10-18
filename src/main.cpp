#include <iostream>
#include <iomanip>
#include <ctime>

#include "OpenCL-Wrapper/opencl.hpp"
#include "OpenCL-Wrapper/kernel.hpp"
#include "Origin/decsol.hpp"
#include "utils.hpp"

#ifndef OPTIMIZATION_LEVEL
#define OPTIMIZATION_LEVEL 0
#endif

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

void log_run(const std::string file_name, const std::size_t size, const double result) {
	std::ofstream outfile;
	outfile.open(file_name, std::ios_base::app);

	auto now = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	outfile << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << " ";

	outfile << std::setw(4) << "-O" << OPTIMIZATION_LEVEL << " ";
	outfile << std::setw(6) << size << " ";
	outfile << std::setw(12) << std::setprecision(6) << std::right << result << std::endl;

	outfile.close(); 
}


std::tuple<std::size_t, std::string, double> parse_result_file(const std::string &s) {
    std::istringstream iss(s);

	std::string word;
	iss >> word; // дата
	iss >> word; // время

	iss >> word; // флаг
	std::string flag = word;

	std::size_t size;
	iss >> size;
	double execution_time;
	iss >> execution_time;

	return std::make_tuple(size, flag, execution_time);
}

std::string average_time(const std::string file_name, const std::size_t current_size) {
	std::ifstream file;
	file.open(file_name);
	if (!file.is_open()) exit(1);

	std::size_t count { 0 };
	double sum_time { 0.0 };
	std::string line;
	std::string current_flag = "-O" + std::to_string(OPTIMIZATION_LEVEL);

	std::vector<double> sample;
	while (std::getline(file, line)) {
		auto [size, flag, execution_time] = parse_result_file(line);
		if (size == current_size && flag == current_flag) {
			count++;
			sum_time += execution_time;
			sample.push_back(execution_time);
		}
	}

	if (count == 0) {
		return "0.0 ± 0.0";
	}

	double avg = sum_time / count;
	double disp { 0 };

	for (std::size_t i = 0; i < count; i++) {
		disp += (avg - sample[i]) * (avg - sample[i]);
	}

	disp /= count - 1;

	return std::to_string(avg) + " ± " + std::to_string(3.0 * std::sqrt(disp));
}

int main() {

	Timer timer {};
	const uint N = 7600u; // size of vectors
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


// ========================
//    Тестовая матрица
// ========================

double** ORIGIN = new double*[N];
for(uint i = 0; i < N; i++) {
	ORIGIN[i] = new double[N];
	for (uint j = 0; j < N; j++)
		ORIGIN[i][j] = drand48() * 1000;
}


#ifdef BENCHMARK
	std::cout << "========================" << std::endl;
	std::cout << "====== Origin dec ======" << std::endl;
	std::cout << "========================" << std::endl << std::endl;

	double** CPU_A = new double*[N];
	int	CPU_ip[N];
	int CPU_ier = 0;

	// initialize memory
	for(std::size_t i = 0; i < N; i++) {
		CPU_A[i] = new double[N];
		for (std::size_t j = 0; j < N; j++)
			CPU_A[i][j] = ORIGIN[i][j];
	}

	// initialize memory
	for(std::size_t i = 0; i < N; i++) {
		CPU_ip[i] = 0; 
	}

	std::cout << "Matrix size = " << N << "x" << N << std::endl;
	std::cout << "Elements number = " << N * N << std::endl;
	std::cout << "Memory = " << (double)(N * N * sizeof(double)) / 1024 / 1024 << " Mb" << std::endl;

	timer.start();
	CPU_ier = dec(N, CPU_A, CPU_ip);
	auto elapsed_time = timer.get();

	if (CPU_ier == 0) {
		log_run("origin_dec.result", N, elapsed_time);
	}

	std::cout << "Current time = " << elapsed_time << " s" << std::endl;
	std::cout << "Average time = " << average_time("origin_dec.result", N) << std::endl;
	// print_vector(N, CPU_ip);

#endif

	std::cout << std::endl;
	std::cout << "========================" << std::endl;
	std::cout << "======== OpenCL ========" << std::endl;
	std::cout << "========================" << std::endl;
	std::cout << std::endl;

	// for (auto device : get_devices()) {
	// 	std::cout << device.name << "\n";
	// }

	Device device(select_device_with_most_flops(), get_opencl_c_code(opencl_c_container())); // compile OpenCL C code for the fastest available device

	Memory<double> GPU_A(device, N * N); // allocate memory on both host and device
	Memory<int>	   GPU_ip(device, N);
	Memory<int>    GPU_ier(device, 1);

#ifdef ORIGIN_TEST
	for(uint n=0u; n<N * N; n++) {
		GPU_A[n] = (n + 1) * 1.0f; // initialize memory
	}

	// Для повышения устойчивости?
	std::swap(GPU_A[N * N - 2], GPU_A[N * N - 1]);
#endif

#ifdef BENCHMARK
	
	for (uint i=0u; i<N; i++) {
		for(uint j=0u; j<N; j++) {
			GPU_A[i * N + j] = ORIGIN[i][j]; // initialize memory
		}
	}

#endif

	for(uint n=0u; n<N; n++) {
		GPU_ip[n] = 1; // initialize memory
	}

	timer.start();

	GPU_A.write_to_device();
	GPU_ip.write_to_device();

	Kernel pivot_kernel(device, "pivot_kernel", get_opencl_c_code(opencl_c_container_pivot_kernel()), 5);
	Kernel    lu_kernel(device,    "lu_kernel", get_opencl_c_code(opencl_c_container()), 5);

	for (uint k = 0; k < N - 1; k++) {
		pivot_kernel.set_parameters(0u, k, N, GPU_A, GPU_ip, GPU_ier);
		   lu_kernel.set_parameters(0u, k, N, GPU_A, GPU_ip, GPU_ier);

		device.enqueue_kernel(pivot_kernel.cl_kernel, 1);
		device.enqueue_kernel(   lu_kernel.cl_kernel, N);
	}

	device.finish_queue();

	GPU_A.read_from_device(); // copy data from device memory to host memory
	GPU_ip.read_from_device();
	GPU_ier.read_from_device();

	elapsed_time = timer.get();

	if (GPU_ier[0] == 0) {
		log_run("gpu_dec.result", N, elapsed_time);
	}

	std::cout << "Current time = " << elapsed_time << " s" << std::endl;
	std::cout << "Average time = " << average_time("gpu_dec.result", N) << std::endl;

	// std::cout << "Spend time = " << elapsed_time << std::endl;
	// std::cout << "Result ier = " << GPU_ier[0] << std::endl;

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

	for (std::size_t i= 0; i < N; i++) {
		if (CPU_ip[i] != GPU_ip[i])
			throw std::runtime_error("Wrong result IP at i = " + std::to_string(i) +
				" Expected: " + std::to_string(CPU_ip[i]) + " but actual is " + std::to_string(GPU_ip[i]));
	}

	for (std::size_t i = 0; i < N; i++) {
		for(std::size_t j = 0; j < N; j++) { 
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

	return EXIT_SUCCESS;
}
