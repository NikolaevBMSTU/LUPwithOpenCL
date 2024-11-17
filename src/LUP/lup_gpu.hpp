#pragma once

#include "../OpenCL-Wrapper/opencl.hpp"
#include "../OpenCL-Wrapper/kernel.hpp"
#include "../OpenCL-Wrapper/utilities.hpp"

#include "../utils.hpp"

#include "kernel.hpp"

// class GPU_LUP_solver {
//     public:
//         GPU_LUP_solver(Device device) :
//             search_kernel(device,    "find_max_in_column_kernel", get_opencl_c_code(find_max_in_column_kernel())),
//             divide_kernel(device,       "identify_column_kernel", get_opencl_c_code(identify_column_kernel())),
//                 lu_kernel(device,       			    "lu_kernel", get_opencl_c_code(lu_kernel_code())),
//               swap_kernel(device,       		      "swap_kernel", get_opencl_c_code(swap_kernel_code())),
//             devide_kernel(device,      		    "devide_kernel", get_opencl_c_code(devide_kernel_code())),
//             forward_kernel(device, "forward_substitution_kernel", get_opencl_c_code(forward_substitution())),
//                back_kernel(device,    "back_substitution_kernel", get_opencl_c_code(back_substitution()))
//                { };


//         int solve(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {

//         }



//     private:
//         Kernel  search_kernel;
//         Kernel  divide_kernel;
//         Kernel      lu_kernel;
//         Kernel    swap_kernel;
//         Kernel  devide_kernel;
//         Kernel forward_kernel;
//         Kernel    back_kernel;
// };

void gpu_lup_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {
    std::cout << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "======== OpenCL ========" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << std::endl;

    // Device device(select_device_with_most_flops()); // create OpenCL device for the fastest available device
    Device device(get_devices()[2]); 
    
    Memory<double> GPU_A(device, N * N); // allocate memory on both host and device
    Memory<double> GPU_b(device, N);
    Memory<int>	   GPU_ip(device, N);
    Memory<int>    GPU_ier(device, 1);

    // initialize memory
    for (std::size_t i = 0; i < N; i++) {
        for(std::size_t j = 0; j < N; j++) {
            GPU_A[i + j * N] = ORIGIN[i][j];
        }

        GPU_b[i] = ORIGIN_VECTOR[i];
        GPU_ip[i] = 1;
    }


    Kernel search_kernel(device,    "find_max_in_column_kernel", get_opencl_c_code(find_max_in_column_kernel()));
    Kernel divide_kernel(device,       "identify_column_kernel", get_opencl_c_code(identify_column_kernel()));
    Kernel     lu_kernel(device,       			    "lu_kernel", get_opencl_c_code(lu_kernel_code()));
    Kernel   swap_kernel(device,       		      "swap_kernel", get_opencl_c_code(swap_kernel_code()));
    Kernel devide_kernel(device,      		    "devide_kernel", get_opencl_c_code(devide_kernel_code()));
    Kernel forward_kernel(device, "forward_substitution_kernel", get_opencl_c_code(forward_substitution()));
    Kernel    back_kernel(device,    "back_substitution_kernel", get_opencl_c_code(back_substitution()));

    println(  "|----------------'------------------------------------------------------------|");

    std::cout << std::endl;
    std::cout << "Matrix size = " << N << "x" << N << std::endl;
    std::cout << "Elements number = " << N * N << std::endl;
    std::cout << "Memory = " << (double)(N * N * sizeof(double)) / 1024 / 1024 << " Mb" << std::endl;

    Timer memory_timer {};

    Timer timer;
    timer.start();

    memory_timer.start();

    GPU_A.write_to_device();
    GPU_b.write_to_device();
    GPU_ip.write_to_device();

    auto memory_to_device_time = memory_timer.get();

    search_kernel.add_parameters(0, (uint)N, GPU_A, GPU_ip, GPU_ier);
    divide_kernel.add_parameters(0, (uint)N, GPU_A, GPU_ip, GPU_ier);
        lu_kernel.add_parameters(0, (uint)N, GPU_A, GPU_ip, GPU_ier);

    // LUP decomposition
    for (uint k = 0; k < N - 1; k++) {
        search_kernel.set_parameters(0, k);
        divide_kernel.set_parameters(0, k);
            lu_kernel.set_parameters(0, k);

        device.enqueue_kernel(search_kernel.cl_kernel, 1);
        device.enqueue_kernel(divide_kernel.cl_kernel, N);
        device.enqueue_kernel(    lu_kernel.cl_kernel, N);
    }

    // прямая подстановка
       swap_kernel.add_parameters(0, GPU_b, GPU_ip);
    forward_kernel.add_parameters(0, (uint)N, GPU_A, GPU_b);

    for (uint k = 0; k < N - 1; k++) {
            swap_kernel.set_parameters(0, k);
        forward_kernel.set_parameters(0, k);
        
        device.enqueue_kernel(swap_kernel.cl_kernel, 1);
        device.enqueue_kernel(forward_kernel.cl_kernel, N);
    }

    // обратная подстановка
    devide_kernel.add_parameters(0, (uint)N, GPU_A, GPU_b);
      back_kernel.add_parameters(0, (uint)N, GPU_A, GPU_b);

    for (uint k = 0; k < N - 1; k++) {
        devide_kernel.set_parameters(0, k);
            back_kernel.set_parameters(0, k);
        
        device.enqueue_kernel(devide_kernel.cl_kernel, 1);
        device.enqueue_kernel(back_kernel.cl_kernel, N);
    }

    device.finish_queue();


    memory_timer.start();

    GPU_A.read_from_device(); // copy data from device memory to host memory
    GPU_b.read_from_device();
    // GPU_ip.enqueue_read_from_device();
    GPU_ier.read_from_device();

    device.finish_queue();

    // Последний шаг
    GPU_b[0] = GPU_b[0] / GPU_A[0 * N + 0];


    auto memory_from_device_time = memory_timer.get();

    auto full_time = timer.get();

    if (GPU_ier[0] == 0) {
        log_run("gpu_dec.result", std::regex_replace(device.info.vendor, std::regex("Intel\\(R\\) Corporation"), "Intel"), N, full_time);
    }

    std::cout << "Current time = " << full_time << " s" << std::endl;
    std::cout << "Average time = " << average_time_string("gpu_dec.result", N,
            std::regex_replace(device.info.vendor, std::regex("Intel\\(R\\) Corporation"), "Intel"), std::to_string(OPTIMIZATION_LEVEL)) << std::endl;
    std::cout << "Copy   to device time = " << memory_to_device_time << " s" << std::endl;
    std::cout << "Copy from device time = " << memory_from_device_time << " s" << std::endl;

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
            res += ORIGIN[i][j] * GPU_b[j];
        }

        if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
                throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
                    " Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
    }

#endif

#ifdef ORIGIN_TEST
    print_matrix(N, N, &A[0]);
    std::cout << std::endl;
    
    print_vector(N, &ip[0]);
    std::cout << "ier = " << ier[0] << std::endl;
#endif
}
