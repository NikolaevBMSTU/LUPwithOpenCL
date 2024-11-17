#pragma once

#include "../OpenCL-Wrapper/opencl.hpp"
#include "../OpenCL-Wrapper/kernel.hpp"
#include "../OpenCL-Wrapper/utilities.hpp"

#include "../utils.hpp"

#include "kernel.hpp"

class GPU_LUP_solver {
    public:
        GPU_LUP_solver(Device &device) :
            device { device },
            search_kernel(this->device,    "find_max_in_column_kernel", get_opencl_c_code(find_max_in_column_kernel())),
            divide_kernel(this->device,       "identify_column_kernel", get_opencl_c_code(identify_column_kernel())),
                lu_kernel(this->device,       			    "lu_kernel", get_opencl_c_code(lu_kernel_code())),
              swap_kernel(this->device,       		      "swap_kernel", get_opencl_c_code(swap_kernel_code())),
            devide_kernel(this->device,      		    "devide_kernel", get_opencl_c_code(devide_kernel_code())),
            forward_kernel(this->device, "forward_substitution_kernel", get_opencl_c_code(forward_substitution())),
               back_kernel(this->device,    "back_substitution_kernel", get_opencl_c_code(back_substitution()))
               { };

        // allocate memory on both host and device
        void allocate_memory(std::size_t N) {
            this->N = N;

              GPU_A = Memory<double>(device, N * N);
              GPU_b = Memory<double>(device, N);
             GPU_ip = Memory<int>(device, N);
            GPU_ier = Memory<int>(device, 1);

            local = Memory<int>(device, device.info.cores);

            search_kernel.add_parameters(0, (uint)N, GPU_A, GPU_ip, GPU_ier);
            divide_kernel.add_parameters(0, (uint)N, GPU_A, GPU_ip, GPU_ier);
                lu_kernel.add_parameters(0, (uint)N, GPU_A, GPU_ip, GPU_ier);

               swap_kernel.add_parameters(0, GPU_b, GPU_ip);
            forward_kernel.add_parameters(0, (uint)N, GPU_A, GPU_b);

            devide_kernel.add_parameters(0, (uint)N, GPU_A, GPU_b);
              back_kernel.add_parameters(0, (uint)N, GPU_A, GPU_b);
        }

        void copy(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {

        }

        void write_to_device() {
            memory_timer.start();

            GPU_A.write_to_device();
            GPU_b.write_to_device();
            GPU_ip.write_to_device();

            memory_to_device_time = memory_timer.get();
        }

        void read_from_device() {
            memory_timer.start();

            GPU_A.read_from_device();
            GPU_b.read_from_device();
            GPU_ip.read_from_device();
            GPU_ier.read_from_device();

            memory_from_device_time = memory_timer.get();
        }

        int solve() {
            write_to_device();

            // LUP decomposition
            std::cout << "LUP decompositon; N = " << N << std::endl;

            for (uint k = 0; k < N - 1; k++) {
                search_kernel.set_parameters(0, k);
                divide_kernel.set_parameters(0, k);
                    lu_kernel.set_parameters(0, k);

                device.enqueue_kernel(search_kernel.cl_kernel, 1);
                device.enqueue_kernel(divide_kernel.cl_kernel, N);
                device.enqueue_kernel(    lu_kernel.cl_kernel, N);
            }

            std::cout << "forward substitution" << std::endl;

            // прямая подстановка
            for (uint k = 0; k < N - 1; k++) {
                   swap_kernel.set_parameters(0, k);
                forward_kernel.set_parameters(0, k);
                
                device.enqueue_kernel(swap_kernel.cl_kernel, 1);
                device.enqueue_kernel(forward_kernel.cl_kernel, N);
            }

            std::cout << "back substitution" << std::endl;

            // обратная подстановка
            for (uint k = 0; k < N - 1; k++) {
                devide_kernel.set_parameters(0, k);
                  back_kernel.set_parameters(0, k);
                
                device.enqueue_kernel(devide_kernel.cl_kernel, 1);
                device.enqueue_kernel(back_kernel.cl_kernel, N - k - 1);
            }

            device.finish_queue();

            read_from_device();

            device.finish_queue();

            // Последний шаг
            GPU_b[0] = GPU_b[0] / GPU_A[0 + 0 * N];

            return GPU_ier[0];
        }

        // allocate memory on both host and device
        Memory<double> GPU_A;
        Memory<double> GPU_b;
        Memory<int>	   GPU_ip;
        Memory<int>    GPU_ier;
        Memory<int> local;

        // Copy to device time, seconds
        double memory_to_device_time;
        // Copy from device time, seconds
        double memory_from_device_time;

    private:
        Device &device;

        Kernel  search_kernel;
        Kernel  divide_kernel;
        Kernel      lu_kernel;
        Kernel    swap_kernel;
        Kernel  devide_kernel;
        Kernel forward_kernel;
        Kernel    back_kernel;

        std::size_t N;

        Timer memory_timer;
        Timer solution_timer;
};

void gpu_lup_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {
    std::cout << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "======== OpenCL ========" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << std::endl;

    // Device device(select_device_with_most_flops()); // create OpenCL device for the fastest available device
    Device device(get_devices()[0]); 

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

    if (ier == 0) {
        log_run("gpu_dec.result", transform_vendor_name(device.info.vendor), N, full_time);
    } else {
        std::cout << "Result code != 0, actual is " << ier << std::endl;
    }

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
            res += ORIGIN[i][j] * GPU_b[j];
        }

        if (std::abs((res - ORIGIN_VECTOR[i]) / ORIGIN_VECTOR[i]) > 1e-6)
                throw std::runtime_error("Wrong solution result at i = " + std::to_string(i) +
                    " Expected: " + std::to_string(ORIGIN_VECTOR[i]) + " but actual is " + std::to_string(res));
    }

#endif

}
