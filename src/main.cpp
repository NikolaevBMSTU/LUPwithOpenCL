#include <iostream>

// #define COMPARE
#define CHECK_SOLUTION

// #define CPU_DECSOL_ORIGIN_BENCHMARK
#define LAPACK_BENCHMARK

// #define CPU_DECSOL_ROW_BENCHMARK
// #define CPU_DECSOL_COLUMN_BENCHMARK

// #define GPU_DECSOL_ROW_BENCHMARK
// #define GPU_DECSOL_COLUMN_BENCHMARK

// #define GPU_VIENNA_BENCHMARK

// #define CPU_BICGSTAB_BENCHMARK
// #define GPU_BICGSTAB_BENCHMARK


#ifdef LAPACK_BENCHMARK
	#include "LAPACK/benchmark.hpp"
#endif

#if defined(GPU_DECSOL_ROW_BENCHMARK) || defined(GPU_DECSOL_COLUMN_BENCHMARK)
	#include "decsol_OpenCL/benchmark.hpp"
#endif

#ifdef GPU_VIENNA_BENCHMARK
	#include "ViennaCL/benchmark.hpp"
#endif

#if defined(CPU_DECSOL_ROW_BENCHMARK) || defined(CPU_DECSOL_COLUMN_BENCHMARK)
	#include "decsol/benchmark.hpp"
#endif

#ifdef CPU_DECSOL_ORIGIN_BENCHMARK
	#include "decsol_origin/benchmark.hpp"
#endif

#if defined(CPU_BICGSTAB_BENCHMARK) || defined(GPU_BICGSTAB_BENCHMARK)
	#include "BiCGStab/benchmark.hpp"
#endif


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

	// ========================
	//    Тестовая матрица
	// ========================

	unsigned short s = 299;
	std::srand(s);
	double** ORIGIN = new double*[N];
	double* ORIGIN_VECTOR = new double[N];
	for(std::size_t i = 0; i < N; i++) {

		ORIGIN[i] = new double[N];
		for (std::size_t j = 0; j < N; j++)
			ORIGIN[i][j] = std::rand() * 1;

		ORIGIN_VECTOR[i] = std::rand() * 1;
	}

#ifdef CPU_DECSOL_ORIGIN_BENCHMARK
	decsol_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef LAPACK_BENCHMARK
	lapack_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef CPU_DECSOL_ROW_BENCHMARK
	decosl_row_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef CPU_DECSOL_COLUMN_BENCHMARK
	decosl_column_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef GPU_DECSOL_COLUMN_BENCHMARK
	gpu_lup_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef GPU_DECSOL_ROW_BENCHMARK
	// gpu_lup_benchmark(N, ORIGIN, ORIGIN_VECTOR);
	throw 1;
#endif

#ifdef GPU_VIENNA_BENCHMARK
	viennacl_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef CPU_BICGSTAB_BENCHMARK
	iml_bicgstab_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

#ifdef GPU_BICGSTAB_BENCHMARK
	viennacl_bicgstab_benchmark(N, ORIGIN, ORIGIN_VECTOR);
#endif

	return EXIT_SUCCESS;
}
