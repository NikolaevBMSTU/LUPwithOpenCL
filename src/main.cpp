#include <iostream>
#include <iomanip>
#include <ctime>

#include "BiCGStab/benchmark.hpp"

#include "LAPACK/lapack.hpp"

// #undef VIENNACL_WITH_OPENCL


#include "viennacl/linalg/lu.hpp"
#include "viennacl/linalg/direct_solve.hpp"

#include "OpenCL-Wrapper/opencl.hpp"
#include "OpenCL-Wrapper/kernel.hpp"
#include "LUP/lup_gpu.hpp"
#include "Origin/decsol.hpp"
#include "Origin_New/decsol.hpp"
#include "utils.hpp"


int main(int argc, char const *argv[]) {
	
	if (argc - 1 > 1) {
		std::cout << "Wrong number of parameters (" << argc - 1 << ")! There must be only one parameter" << std::endl;
		return EXIT_FAILURE;
	}

	if (argc - 1 == 0) {
		std::cout << "Wrong number of parameters (" << argc - 1 << ")! You must set Matrix size" << std::endl;
		return EXIT_FAILURE;
	}

	const std::size_t N = std::atoi(argv[1]); // size of vectors

	if (N == 0) {
		std::cout << "N cannot be 0" << std::endl;
		return EXIT_FAILURE;
	}

	#undef ORIGIN_TEST
	// #define COMPARE
	#define CHECK_SOLUTION
	#define BENCHMARK
	#define LAPACK_BENCHMARK
	// #define CPU_DECSOL_BENCHMARK
	#define CPU_DECSOL_ROW_BENCHMARK
	// #define CPU_DECSOL_COLUMN_BENCHMARK
	#define GPU_DECSOL_COLUMN_BENCHMARK
	// #define GPU_DECSOL_ROW_BENCHMARK
	// #define GPU_VIENNA_BENCHMARK
	// #define GPU_BICGSTAB_BENCHMARK
	// #define CPU_BICGSTAB_BENCHMARK

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

	unsigned short s = 299;
	std::srand(s);
	double** ORIGIN = new double*[N];
	double* ORIGIN_VECTOR = new double[N];
	for(uint i = 0; i < N; i++) {

		ORIGIN[i] = new double[N];
		for (uint j = 0; j < N; j++)
			ORIGIN[i][j] = std::rand() * 1000;

		ORIGIN_VECTOR[i] = std::rand() * 1000;
	}

#ifdef CPU_DECSOL_BENCHMARK
	// {
		std::cout << std::endl;
		std::cout << "========================" << std::endl;
		std::cout << "====== Origin dec ======" << std::endl;
		std::cout << "========================" << std::endl << std::endl;

		double** CPU_A = new double*[N];
		double*  CPU_b = new double[N];
		int*	 CPU_ip = new int[N];
		int CPU_ier = 0;

		// initialize memory
		for(std::size_t i = 0; i < N; i++) {
			CPU_A[i] = new double[N];
			for (std::size_t j = 0; j < N; j++)
				CPU_A[i][j] = ORIGIN[i][j];

			CPU_b[i] = ORIGIN_VECTOR[i];
			CPU_ip[i] = 0;
		}

		std::cout << "Matrix size = " << N << "x" << N << std::endl;
		std::cout << "Elements number = " << N * N << std::endl;
		std::cout << "Memory = " << (double)(N * (N + 1) * sizeof(double) + (2 * N + 1) * sizeof(int))  / 1024 / 1024 << " Mb" << std::endl;

		Timer timer {};
		timer.start();
		CPU_ier = dec(N, CPU_A, CPU_ip);
		if (CPU_ier != 0) {
			throw std::runtime_error("Solution was not successful. Error code " + std::to_string(CPU_ier));
		}
		sol(N, CPU_A, CPU_b, CPU_ip);
		auto elapsed_time = timer.get();

		std::cout << "Current time = " << elapsed_time << " s" << std::endl;
		std::cout << "Average time = " << average_time_string("origin_dec.result", N, "CPU", std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

		// print_matrix(N, N, ORIGIN);
		// print_vector(N, ORIGIN_VECTOR);
		// print_vector(N, CPU_b);

#ifdef CHECK_SOLUTION

		for (std::size_t i = 0; i < N; i++) {

			double res { 0 };

			for (std::size_t j = 0; j < N; j++) {
				res += ORIGIN[i][j] * CPU_b[j];
			}

			if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
					throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
						" Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
		}

#endif

		log_run("origin_dec.result", "CPU", N, elapsed_time);

	// }
#endif

#ifdef LAPACK_BENCHMARK
	lapack_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef CPU_DECSOL_ROW_BENCHMARK
	{
		std::cout << std::endl;
		std::cout << "========================" << std::endl;
		std::cout << "=== Origin row decsol ==" << std::endl;
		std::cout << "========================" << std::endl << std::endl;

		double* CPU_A  = new double[N * N];
		double* CPU_b  = new double[N];
		int*	CPU_ip = new int[N];
		int CPU_ier = 0;

		// initialize memory
		for(std::size_t i = 0; i < N; i++) {
			for (std::size_t j = 0; j < N; j++)
				CPU_A[i + N * j] = ORIGIN[i][j]; // модель хранения по строкам

			CPU_b[i] = ORIGIN_VECTOR[i];
			CPU_ip[i] = 0;
		}

		std::cout << "Matrix size = " << N << "x" << N << std::endl;
		std::cout << "Elements number = " << N * N << std::endl;
		std::cout << "Memory = " << (double)(N * (N + 1) * sizeof(double) + (2 * N + 1) * sizeof(int))  / 1024 / 1024 << " Mb" << std::endl;

		Timer timer {};
		timer.start();
		CPU_ier = dec_row(N, CPU_A, CPU_ip);
		if (CPU_ier != 0) {
			throw std::runtime_error("Solution was not successful. Error code " + std::to_string(CPU_ier));
		}
		sol_row(N, CPU_A, CPU_b, CPU_ip);
		auto elapsed_time = timer.get();

		std::cout << "Current time = " << elapsed_time << " s" << std::endl;
		std::cout << "Average time = " << average_time_string("origin_dec.result", N, "CPU", std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

		// print_matrix(N, N, ORIGIN);
		// print_vector(N, ORIGIN_VECTOR);
		// print_vector(N, CPU_b);

#ifdef CHECK_SOLUTION

		for (std::size_t i = 0; i < N; i++) {

			double res { 0 };

			for (std::size_t j = 0; j < N; j++) {
				res += ORIGIN[i][j] * CPU_b[j];
			}

			if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
					throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
						" Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
		}

#endif

		log_run("origin_dec.result", "CPU", N, elapsed_time);

	}
#endif

#ifdef CPU_DECSOL_COLUMN_BENCHMARK
	{
		std::cout << std::endl;
		std::cout << "========================" << std::endl;
		std::cout << "== Originj column dec ==" << std::endl;
		std::cout << "========================" << std::endl << std::endl;

		double* CPU_A = new double[N * N];
		double*  CPU_b = new double[N];
		int*	 CPU_ip = new int[N];
		int CPU_ier = 0;

		// initialize memory
		for(std::size_t i = 0; i < N; i++) {
			for (std::size_t j = 0; j < N; j++)
				CPU_A[i * N + j] = ORIGIN[i][j]; // модель хранения по столбцам

			CPU_b[i] = ORIGIN_VECTOR[i];
			CPU_ip[i] = 0;
		}

		std::cout << "Matrix size = " << N << "x" << N << std::endl;
		std::cout << "Elements number = " << N * N << std::endl;
		std::cout << "Memory = " << (double)(N * (N + 1) * sizeof(double) + (2 * N + 1) * sizeof(int))  / 1024 / 1024 << " Mb" << std::endl;

		Timer timer {};
		timer.start();
		CPU_ier = dec_column(N, CPU_A, CPU_ip);
		if (CPU_ier != 0) {
			throw std::runtime_error("Solution was not successful. Error code " + std::to_string(CPU_ier));
		}
		sol_column(N, CPU_A, CPU_b, CPU_ip);
		auto elapsed_time = timer.get();

		std::cout << "Current time = " << elapsed_time << " s" << std::endl;
		std::cout << "Average time = " << average_time_string("origin_dec.result", N, "CPU", std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

		// print_matrix(N, N, ORIGIN);
		// print_vector(N, ORIGIN_VECTOR);
		// print_vector(N, CPU_b);

#ifdef CHECK_SOLUTION

		for (std::size_t i = 0; i < N; i++) {

			double res { 0 };

			for (std::size_t j = 0; j < N; j++) {
				res += ORIGIN[i][j] * CPU_b[j];
			}

			if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
					throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
						" Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
		}

#endif

		log_run("origin_dec.result", "CPU", N, elapsed_time);

	}
#endif


#ifdef GPU_DECSOL_COLUMN_BENCHMARK
	gpu_lup_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif // GPU_DECSOL_COLUMN_BENCHMARK

#ifdef GPU_DECSOL_ROW_BENCHMARK
	// gpu_lup_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif // GPU_DECSOL_ROW_BENCHMARK

#ifdef GPU_VIENNA_BENCHMARK
	{
		std::cout << std::endl;
		std::cout << "========================" << std::endl;
		std::cout << "======= ViennaCL =======" << std::endl;
		std::cout << "========================" << std::endl;
		std::cout << std::endl;

		viennacl::matrix<double, viennacl::row_major> vcl_A(N, N);

		viennacl::vector<double> vcl_vec_B(N);

		for (std::size_t i = 0; i < N; i++) {
			for(std::size_t j = 0; j < N; j++) {
				vcl_A(i, j) = ORIGIN[i][j]; // initialize memory
			}

			vcl_vec_B(i) = ORIGIN_VECTOR[i];
		}

		Timer timer;
		timer.start();

		viennacl::linalg::lu_factorize(vcl_A);
		viennacl::linalg::lu_substitute(vcl_A, vcl_vec_B);

		auto& vcl_vec_result = vcl_vec_B;

		timer.get();

		auto full_time = timer.get();

		std::cout << viennacl::ocl::current_device().info() << std::endl;

		std::cout << "Current time = " << full_time << " s" << std::endl;
		std::cout << "Average time = " << average_time_string("gpu_vienna.result", N,
				std::regex_replace(viennacl::ocl::current_device().vendor(), std::regex("Intel\\(R\\) Corporation"), "Intel"), std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

	#ifdef CHECK_SOLUTION

		// for (std::size_t i = 0; i < N; i++) {
		// 	std::cout << vcl_vec_result(i) << " ";
		// }

		// std::cout << std::endl;

		// print_vector(N, &GPU_b[0]);

		for (std::size_t i = 0; i < N; i++) {

			double res { 0 };

			for (std::size_t j = 0; j < N; j++) {
				res += ORIGIN[i][j] * vcl_vec_result(j);
			}

			if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-3)
					throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
						" Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
		}

	#endif

		log_run("gpu_vienna.result", std::regex_replace(viennacl::ocl::current_device().vendor(), std::regex("Intel\\(R\\) Corporation"), "Intel"), N, full_time);

	}
#endif // GPU_VIENNA_BENCHMARK

#ifdef CPU_BICGSTAB_BENCHMARK
	iml_bicgstab_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif // CPU_BICGSTAB_BENCHMARK

#ifdef GPU_BICGSTAB_BENCHMARK
	viennacl_bicgstab_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif // GPU_BICGSTAB_BENCHMARK

	return EXIT_SUCCESS;
}
