#include <lapacke.h>

#include "../utils.hpp"

void lapack_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {
    
    std::cout << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "======= LAPACKE =======" << std::endl;
    std::cout << "=======================" << std::endl << std::endl;

    std::unique_ptr<double[]> CPU_A = std::make_unique<double[]>(N * N);
    // double* CPU_A  = new double[N * N];
    double* CPU_b  = new double[N];
    int*	CPU_ip = new int[N];
    int CPU_ier = 0;

    // initialize memory
    for(std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < N; j++)
            CPU_A[i + N * j] = ORIGIN[i][j]; // модель хранения по столбцам

        CPU_b[i] = ORIGIN_VECTOR[i];
        CPU_ip[i] = i;
    }

    std::cout << "Matrix size = " << N << "x" << N << std::endl;
    std::cout << "Elements number = " << N * N << std::endl;
    std::cout << "Memory = " << (double)(N * (N + 1) * sizeof(double) + (2 * N + 1) * sizeof(int))  / 1024 / 1024 << " Mb" << std::endl;

    Timer timer {};

    timer.start();
    CPU_ier = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, CPU_A.get(), N, CPU_ip, CPU_b, 1);
    if (CPU_ier != 0) {
        throw std::runtime_error("Solution was not successful. Error code " + std::to_string(CPU_ier));
    }
    auto elapsed_time = timer.get();

    std::cout << "Current time = " << elapsed_time << " s" << std::endl;
    std::cout << "Average time = " << average_time_string("lapack.result", N, "CPU", std::to_string(OPTIMIZATION_LEVEL)) << std::endl;

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

    log_run("lapack.result", "CPU", N, elapsed_time);

}
