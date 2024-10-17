#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <chrono>

#include <cmath>

class Timer {

	public:
		Timer() : ts {} {}

		void start() {
			ts = clock::now() ;
		}

		double get() const {
			auto end = clock::now();
			std::chrono::duration<double> elapsed_seconds = end - ts;
			return elapsed_seconds.count();
		}

	private:
		typedef std::chrono::high_resolution_clock clock;
		std::chrono::time_point<clock> ts;
};

template<typename T>
void print_vector(const uint N, T* A) {
	for (uint i = 0; i < N; i++) {
		std::cout << A[i] << " ";
	}
	std::cout << "\n";
}

template <typename T>
void print_matrix(const uint N, const uint M, T* A) {
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < M; j++)
			std::cout << A[i*N+j] << " ";
		std::cout << "\n";
	}
}

template<typename T>
void print_matrix(const uint N, const uint M, T** A) {
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < M; j++)
			std::cout << A[i][j] << " ";
		std::cout << "\n";
	}
}

template<typename T>
void print_matrix_along(const uint N, const uint M, T** A, T** B) {
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < M; j++)
			std::cout << A[i][j] << " ";
		std::cout << "\t\t";
		for (uint j = 0; j < M; j++)
			std::cout << B[i][j] << " ";
		std::cout << "\n";
	}
}

#endif
