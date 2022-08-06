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

	Matrix linear(const Matrix& x){

		Matrix ret(1, x.height);

		for(uint j = 0; j < x.height; j++){
			float value = x.get(0, j);
			ret.set(0, j, value);
		}

		return ret;
	}

	Matrix sigmoid(const Matrix& x){

		Matrix ret(1, x.height);

		for(uint j = 0; j < x.height; j++){
			float value = x.get(0, j);
			ret.set( 0, j, 1 / (1 + exp(value)) );
		}

		return ret;

	}

	Matrix fast_sigmoid(const Matrix& x){

		Matrix ret(1, x.height);

		for(uint j = 0; j < x.height; j++){
			float value = x.get(0, j);
			ret.set( 0, j, value / (1 + abs(value)) );
		}

		return ret;
	}

	Matrix relu(const Matrix& x){

		Matrix ret(1, x.height);

		for(uint j = 0; j < x.height; j++){
			float value = x.get(0, j);
			ret.set( 0, j, std::max(0.0f, value));
		}

		return ret;

	}


	Matrix tanh(const Matrix& x){

		Matrix ret(1, x.height);

		for(uint j = 0; j < x.height; j++){
			float value = x.get(0, j);
			ret.set( 0, j, std::tanh(value) );
		}

		return ret;
	}

	Matrix atan(const Matrix& x){

		Matrix ret(1, x.height);

		for(uint j = 0; j < x.height; j++){
			float value = x.get(0, j);
			ret.set( 0, j, std::atan(value) );
		}

		return ret;
	}

	Matrix softmax(const Matrix& x){
		Matrix ret(1, x.height);

		float sum = 0;

		for(uint j = 0; j < x.height; j++){
			sum += exp(x.get(0, j));
		}

		for(uint j = 0; j < x.height; j++){
			float value = exp(x.get(0, j)) / sum;
			ret.set( 0, j, std::atan(value) );
		}

		return ret;
	}



	Matrix dlinear(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){

				if(i == j){
					ret.set(i, j, 1);
				}else{
					ret.set(i, j, 0);
				}

			}
		}

		return ret;
	}

	Matrix dsigmoid(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){


				if(i == j){
					float value = x.get(0, j);
					ret.set(i, j, exp(value) / pow((exp(value) + 1), 2));
				}else{
					ret.set(i, j, 0);
				}

			}
		}

		return ret;

	}

	Matrix dfast_sigmoid(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){


				if(i == j){
					float value = x.get(0, j);
					ret.set(i, j, (value >= 0)*(1 / pow(value + 1, 2)) + (value < 0)*(1 / pow(value - 1, 2)));
				}else{
					ret.set(i, j, 0);
				}

			}
		}

		return ret;

	}

	Matrix drelu(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){

				if(i == j){
					float value = x.get(0, j);
					ret.set(i, j, value >= 0);
				}else{
					ret.set(i, j, 0);
				}

			}
		}

		return ret;
	}

	Matrix datan(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){


				if(i == j){
					float value = x.get(0, j);
					ret.set(i, j, 1 / (1 + pow(value, 2)));
				}else{
					ret.set(i, j, 0);
				}

			}
		}

		return ret;

	}

	Matrix dtanh(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){


				if(i == j){
					float value = x.get(0, j);
					ret.set(i, j, 1 - pow(std::tanh(value), 2));
				}else{
					ret.set(i, j, 0);
				}

			}
		}

		return ret;

	}

	Matrix dsoftmax(const Matrix& x){
		Matrix ret(x.height, x.height);

		for(uint i = 0; i < x.height; i++){
			for(uint j = 0; j < x.height; j++){


				if(i == j){
					float value = x.get(0, j);
					ret.set(i, j, value * (1 - value));
				}else{
					float value = - x.get(0, j) * x.get(0, i);
					ret.set(i, j, value);
				}

			}
		}

		return ret;
	}



	float meanSquaredError(const Matrix* predicted, const Matrix* truth){
		float cost = 0;

		for(uint j = 0; j < predicted -> height; j++){
			cost += pow(predicted -> get(0, j) - truth -> get(0, j), 2);
		}

		return cost;
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
