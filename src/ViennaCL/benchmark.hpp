// #include 
#ifndef VIENNACL_BENCHMARK_H
#define VIENNACL_BENCHMARK_H

#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/backend.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/scalar.hpp"

#include "viennacl/linalg/lu.hpp"
#include "viennacl/linalg/direct_solve.hpp"

#include "../utils.hpp"

void viennacl_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {
    
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
    std::cout << "Average time = " << average_time_string("gpu_vienna.result", N, transform_vendor_name(viennacl::ocl::current_device().vendor()),
                std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

#ifdef CHECK_SOLUTION

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

		log_run("gpu_vienna.result", transform_vendor_name(viennacl::ocl::current_device().vendor()), N, full_time);

}

#endif
