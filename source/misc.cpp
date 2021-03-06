#include "../include/misc.h"
#include <random>
#include <cmath>

namespace nnlib {

	float random() {

		std::random_device rd{};
		std::mt19937 engine{rd()};
		std::uniform_real_distribution<double> dist{0.0, 1.0};
		double ret = dist(engine);
		return ret;

	}

	double fast_pow(double base, uint exponent) {
		double result = 1;
		while (exponent) {
			if (!(exponent % 2)) {
				result *= base;
			}
			base *= base;
			exponent >>= 1; // divide by 2
		}
		return result;
	}

	float random(float min, float max) {
		return random()*(max-min) + min;
	}

	int randomInt(int min, int max){
		int ret = (int) random((float)min, (float)max + 1);
		return ret - (ret > max); //dont allow numbers bigger than max
	}

	int numlen(int x) {
		return floor(log10(x)) + 1;
	}

	float linear(float x){
		return x;
	}

	float sigmoid(float x){
		return 1 / (1 + exp(-x));
	}

	float fast_sigmoid(float x){
		return x / (1 + abs(x));
	}

	float relu(float x){
		return std::max(0.0f, x);
	}




	std::vector<std::string> splitString(std::string string, std::string split_by) {

		std::vector<std::string> ret;

		while (string.find(split_by) != std::string::npos) {
			int spacepos = string.find(split_by);
			ret.push_back(string.substr(0, spacepos));
			string = string.substr(spacepos + split_by.size(), string.size());
		}

		ret.push_back(string.substr(0, string.find(split_by)));

		return ret;

	}

	uint getProcessorCount(){
		const uint processors = std::thread::hardware_concurrency();
		if(processors == 0)
			return 1;

		return processors;
	}

}
