#ifndef DECSOL_ORIGIN_H
#define DECSOL_ORIGIN_H

#include <iostream>

#include "decsol.hpp"

#include "../utils.hpp"

void decsol_benchmark(std::size_t N, double** ORIGIN, double* ORIGIN_VECTOR) {
    
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