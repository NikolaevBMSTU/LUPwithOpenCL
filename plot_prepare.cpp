#include <fstream>
#include <iomanip>
#include <string>

#include <vector>
#include <unordered_map>
#include <set>

#include "utils.hpp"

void prepare(std::string input_file, std::string output_file, std::string vendor_name) {
	std::ifstream input;

	input.open(input_file);
	if (!input.is_open()) exit(1);

	std::ofstream output;
	output.open(output_file);
	if (!output.is_open()) exit(1);

	std::set<int> sizes {};

	std::string line;
	while (std::getline(input, line)) {
		auto [size, flag, vendor, execution_time] = parse_result_file(line);
		sizes.emplace(size);
	}
	
	std::unordered_map<int, std::tuple<double, double>> times {};
	for (auto size : sizes) {
		times.emplace(size, average_time(input_file, size, vendor_name, "3"));
	}

	for (auto [size, res] : times) {
		auto [time, error] = res;
		if (time < 1e-6) output << "#";
		output << std::to_string(size) << "\t" << std::to_string(time) << "\t" << std::to_string(error) << "\n";
	}

	input.close();
	output.close();
}

int main() {
	prepare("origin_dec.result", "cpu.result", "CPU");
	prepare("gpu_dec.result", "intel.result", "Intel");
	prepare("gpu_dec.result", "amd.result", "AMD");
}
