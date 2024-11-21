#ifndef BICGSTAB_BENCHMARK
#define BICGSTAB_BENCHMARK

#include <iostream>

#include "iml.hpp"
#include "viennacl.hpp"

#include "../utils.hpp"

void iml_bicgstab_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {

	std::cout << std::endl;
	std::cout << "=========================" << std::endl;
	std::cout << "===== IML BiCGStab =====" << std::endl;
	std::cout << "=========================" << std::endl << std::endl;

	std::cout << "Matrix size = " << N << "x" << N << std::endl;
	std::cout << "Elements number = " << N * N << std::endl;
	std::cout << "Memory = " << (double)(N * N * sizeof(double)) / 1024 / 1024 << " Mb" << std::endl;

	ublas::matrix<double> CPU_A(N, N);
	ublas::vector<double> CPU_rhs(N);
	ublas::vector<double> CPU_result(N);

    // initialize memory
	for (std::size_t i = 0; i < N; i++) {
		for(std::size_t j = 0; j < N; j++) {
			CPU_A(i,j) = ORIGIN[i][j];
		}

		CPU_rhs[i] = ORIGIN_VECTOR[i];
		CPU_result[i] = 1;
	}
	
    Timer timer;
	timer.start();

	auto [solved, iter, resudial] = BiCGSTAB(CPU_A, CPU_result, CPU_rhs, 1000, 1e-6);

	std::cout << "Return code " << solved << std::endl;
	std::cout << "Final resudial " << resudial << std::endl;
	std::cout << "Performed iterations " << iter << std::endl;

	auto full_time = timer.get();

	std::cout << "Current time = " << full_time << " s" << std::endl;
	std::cout << "Average time = " << average_time_string("cpu_bicgstab.result", N, "CPU", std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

	std::cout << std::endl;

#ifdef COMPARE

	// for (uint i = 0; i < N; i++) {
	// 	std::cout << GPU_result[i] << " ";
	// }
	// std::cout << "\n";

	print_vector(N, &CPU_b[0]);
	print_vector(N, &CPU_result[0]);

	// for (std::size_t i = 0; i < N; i++) {
	// 	if (std::abs((CPU_b[i] - GPU_result_on_host[i]) / CPU_b[i]) > 1e-6)
	// 		throw std::runtime_error("Wrong result LU at i = " + std::to_string(i) +
	// 			" Expected: " + std::to_string(CPU_b[i]) + " but actual is " + std::to_string(GPU_result_on_host[i]));
	// }

#endif

#ifdef CHECK_SOLUTION

	for (std::size_t i = 0; i < N; i++) {

		double res { 0 };

		for (std::size_t j = 0; j < N; j++) {
			res += ORIGIN[i][j] * CPU_result[j];
		}

		if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
				throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
					" Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
	}

#endif

	log_run("cpu_bicgstab.result", "CPU", N, full_time);

}

#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/backend.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/scalar.hpp"

#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/ilu.hpp"

void viennacl_bicgstab_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {

    std::cout << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "=== ViennaCL BiCGStab ===" << std::endl;
    std::cout << "=========================" << std::endl << std::endl;

    viennacl::ocl::set_context_platform_index(0, 0);
    viennacl::ocl::device device = viennacl::ocl::current_device();
    std::cout << "Vienna device vendor " << device.vendor() << std::endl;
    std::cout << "Vienna device name " << device.name() << std::endl << std::endl;

    std::cout << "Matrix size = " << N << "x" << N << std::endl;
    std::cout << "Elements number = " << N * N << std::endl;
    std::cout << "Memory = " << (double)(N * N * sizeof(double)) / 1024 / 1024 << " Mb" << std::endl;

    viennacl::matrix<double> GPU_A(N, N);
    viennacl::vector<double> GPU_b(N);
    viennacl::vector<double> GPU_result(N);

    Timer timer;
    timer.start();

    class ArrayWrapper {
        public:
            double** a;
            std::size_t size;
            explicit ArrayWrapper(std::size_t N, double** array) : size { N }, a { array } {};
            std::size_t size1() const { return size; };
            std::size_t size2() const { return size; };
            double operator()(const std::size_t& i, const std::size_t& j) const {
                return a[i][j];
            }
    } array(N, ORIGIN);

    Timer memory_timer {};
    memory_timer.start();
    copy(array, GPU_A);
    copy(std::vector<double>(ORIGIN_VECTOR, ORIGIN_VECTOR + N), GPU_b);

    // for (std::size_t i = 0; i < N; i++) {
    // 	for(std::size_t j = 0; j < N; j++) {
    // 		GPU_A(i,j) = ORIGIN[i][j]; // initialize memory
    // 	}

    // 	GPU_b[i] = ORIGIN_VECTOR[i];
    // }


    auto memory_to_device_time = memory_timer.get();

    std::cout << "Начинаем" << std::endl;

    auto solver = viennacl::linalg::bicgstab_tag{1e-6, 10000};
    GPU_result = viennacl::linalg::solve(GPU_A, GPU_b, solver); 

    std::cout << "Relative error " << solver.error() << std::endl;
    std::cout << "Performed iterations " << solver.iters() << std::endl;

    viennacl::backend::finish();

    memory_timer.start();
    std::vector<double> GPU_result_on_host(N);
    copy(GPU_result, GPU_result_on_host);
    auto memory_from_device_time = memory_timer.get();

    auto full_time = timer.get();

    std::cout << "Current time = " << full_time << " s" << std::endl;
    std::cout << "Average time = " << average_time_string("gpu_bicgstab.result", N,
            std::regex_replace(device.vendor(), std::regex("Intel\\(R\\) Corporation"), "Intel"), std::to_string(OPTIMIZATION_LEVEL)) << std::endl;
    std::cout << "Copy   to device time = " << memory_to_device_time << " s" << std::endl;
    std::cout << "Copy from device time = " << memory_from_device_time << " s" << std::endl;

    std::cout << std::endl;

#ifdef COMPARE

    // for (uint i = 0; i < N; i++) {
    // 	std::cout << GPU_result[i] << " ";
    // }
    // std::cout << "\n";

    // print_vector(N, &GPU_result[0]);
    // print_vector(N, &GPU_result_on_host[0]);

    for (std::size_t i = 0; i < N; i++) {
        if (std::abs((CPU_b[i] - GPU_result_on_host[i]) / CPU_b[i]) > 1e-6)
            throw std::runtime_error("Wrong result LU at i = " + std::to_string(i) +
                " Expected: " + std::to_string(CPU_b[i]) + " but actual is " + std::to_string(GPU_result_on_host[i]));
    }

#endif

#ifdef CHECK_SOLUTION

    for (std::size_t i = 0; i < N; i++) {

        double res { 0 };

        for (std::size_t j = 0; j < N; j++) {
            res += ORIGIN[i][j] * GPU_result_on_host[j];
        }

        if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
                throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
                    " Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
    }

#endif

    log_run("gpu_bicgstab.result", transform_vendor_name(device.name()), N, full_time);

}

#endif
