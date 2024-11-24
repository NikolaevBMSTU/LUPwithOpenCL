#ifndef DECSOL_GPU_H
#define DECSOL_GPU_H

#include <iostream>

#include "../OpenCL-Wrapper/opencl.hpp"
#include "lup_gpu.hpp"

#include "../utils.hpp"

void gpu_lup_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {

    std::cout << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "======== OpenCL ========" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << std::endl;

    // Device device(select_device_with_most_flops()); // create OpenCL device for the fastest available device
    Device device(get_devices()[1]); 

    GPU_LUP_solver solver(device);
    solver.allocate_memory(N);

    // initialize memory
    for (std::size_t i = 0; i < N; i++) {
        for(std::size_t j = 0; j < N; j++) {
            solver.GPU_A[i + j * N] = ORIGIN[i][j];
        }

        solver.GPU_b[i] = ORIGIN_VECTOR[i];
        solver.GPU_ip[i] = 1;
    }

    println(  "|----------------'------------------------------------------------------------|");

    std::cout << std::endl;
    std::cout << "Matrix size = " << N << "x" << N << std::endl;
    std::cout << "Elements number = " << N * N << std::endl;
    std::cout << "Memory = " << (double)(N * N * sizeof(double)) / 1024 / 1024 << " Mb" << std::endl;

    Timer timer;
    timer.start();

    auto ier = solver.solve();    

    auto full_time = timer.get();

    std::cout << "Current time = " << full_time << " s" << std::endl;
    std::cout << "Average time = " << average_time_string("gpu_dec.result", N, transform_vendor_name(device.info.vendor), std::to_string(OPTIMIZATION_LEVEL)) << std::endl;
    std::cout << "Copy   to device time = " << solver.memory_to_device_time << " s" << std::endl;
    std::cout << "Copy from device time = " << solver.memory_from_device_time << " s" << std::endl;

    std::cout << std::endl;

    // std::cout << "Сравнение CPU_ip и GPU_ip\n";
    // print_vector(N, CPU_ip);
    // print_vector(N, GPU_ip.data());

    // print_matrix(3, N, &GPU_A[0]);


#ifdef CHECK_SOLUTION

    // print_vector(N, &CPU_b[0]);
    // print_vector(N, &GPU_b[0]);

    for (std::size_t i = 0; i < N; i++) {

        double res { 0 };

        for (std::size_t j = 0; j < N; j++) {
            res += ORIGIN[i][j] * solver.GPU_b[j];
        }

        if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
                throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
                    " Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
    }

#endif

    if (ier == 0) {
        log_run("gpu_dec.result", transform_vendor_name(device.info.vendor), N, full_time);
    } else {
        std::cout << "Result code != 0, actual is " << ier << std::endl;
    }

}

#endif
