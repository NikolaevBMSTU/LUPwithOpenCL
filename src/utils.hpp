#ifndef UTILS_H
#define UTILS_H

#ifndef OPTIMIZATION_LEVEL
#define OPTIMIZATION_LEVEL 0
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <tuple>
#include <vector>
#include <regex>

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
void print_vector(const std::size_t N, T* A) {
	for (std::size_t i = 0; i < N; i++) {
		std::cout << A[i] << " ";
	}
	std::cout << "\n";
}

template <typename T>
void print_matrix(const std::size_t N, const std::size_t M, T* A) {
	for (std::size_t i = 0; i < N; i++) {
		for (std::size_t j = 0; j < M; j++)
			std::cout << A[i*N+j] << " ";
		std::cout << "\n";
	}
}

template<typename T>
void print_matrix(const std::size_t N, const std::size_t M, T** A) {
	for (std::size_t i = 0; i < N; i++) {
		for (std::size_t j = 0; j < M; j++)
			std::cout << A[i][j] << " ";
		std::cout << "\n";
	}
}

template<typename T>
void print_matrix_along(const std::size_t N, const std::size_t M, T** A, T** B) {
	for (std::size_t i = 0; i < N; i++) {
		for (std::size_t j = 0; j < M; j++)
			std::cout << A[i][j] << " ";
		std::cout << "\t\t";
		for (std::size_t j = 0; j < M; j++)
			std::cout << B[i][j] << " ";
		std::cout << "\n";
	}
}

void log_run(const std::string file_name, const std::string vendor, const std::size_t size, const double result) {
	std::ofstream outfile;
	outfile.open(file_name, std::fstream::in | std::fstream::out | std::fstream::app);

	if (!outfile) {
		outfile.open(file_name, std::fstream::in | std::fstream::out | std::fstream::trunc);
	}

	auto now = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	outfile << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << " ";

	outfile << std::setw(4) << "-O" << OPTIMIZATION_LEVEL << " ";
	outfile << std::setw(6) << size << " ";
	outfile << std::setw(12) << std::setprecision(6) << std::right << result;
	outfile << std::setw(7) << vendor << std::endl;

	outfile.close(); 
}

std::tuple<std::size_t, std::string, std::string, double> parse_result_file(const std::string &s) {
    std::istringstream iss(s);

	std::string word;
	iss >> word; // дата

	if (word[0] == '0') return std::make_tuple(-1, "", "", 0);

	iss >> word; // время

	iss >> word; // флаг
	std::string flag = word;

	std::size_t size;
	iss >> size;
	double execution_time;
	iss >> execution_time;
	std::string vendor;
	iss >> vendor;

	return std::make_tuple(size, flag, vendor, execution_time);
}

std::tuple<double, double> average_time(const std::string file_name, const std::size_t current_size,
		const std::string vendor_name, const std::string optimization_level) {
	std::ifstream file;
	file.open(file_name);
	if (!file.is_open()) return {0.0, 0.0};

	std::size_t count { 0 };
	double sum_time { 0.0 };
	std::string line;
	std::string current_flag = "-O" + optimization_level;

	std::vector<double> sample;
	while (std::getline(file, line)) {
		auto [size, flag, vendor, execution_time] = parse_result_file(line);
		if (size == current_size && flag == current_flag && vendor == vendor_name) {
			count++;
			sum_time += execution_time;
			sample.push_back(execution_time);
		}
	}

	if (count == 0) {
		return std::make_tuple(0.0, 0.0);
	}

	double avg = sum_time / count;
	double disp { 0 };

	for (std::size_t i = 0; i < count; i++) {
		disp += (avg - sample[i]) * (avg - sample[i]);
	}

	disp /= count - 1;

	disp = std::sqrt(disp);

	return std::make_tuple(avg, 3.0 * disp);
}

std::string average_time_string(const std::string file_name, const std::size_t current_size,
		const std::string vendor_name, const std::string optimization_level) {

	auto [time, error] = average_time(file_name, current_size, vendor_name, optimization_level);
	return std::to_string(time) + " ± " + std::to_string(error);
}

std::string transform_vendor_name(const std::string vendor_name) {
	auto tmp = vendor_name;
	tmp = std::regex_replace(tmp, std::regex("Intel\\(R\\) Corporation"), "Intel");
	tmp = std::regex_replace(tmp, std::regex("Advanced Micro Devices, Inc."), "AMD");
	return tmp;
}

#endif
