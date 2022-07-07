#include "matrix.h"
//is this double declaration of <vector> and <string>?
#include <vector>
#include <string>
#include <thread>

namespace nnlib {

	// uniformly distributed random float between 0 and 1
	float random();
	float random(float min, float max);
	int randomInt(int min, int max);

	double fast_pow(double base, uint exponent);

	// number of digits x takes up when printed (numlen(0) = 1)
	int numlen(int x);


	// activation functions
	float linear(float x);

	float sigmoid(float x);

	float fast_sigmoid(float x);

	float relu(float x);

	float sparseCategoricalCrossentropy(Matrix* predicted, uint truth);

	float categoricalCrossentropy(Matrix* predicted, Matrix* truth);

	float binaryCrossentropy(Matrix* predicted, Matrix* truth);

	float meanSquaredError(Matrix* predicted, Matrix* truth);


	std::vector<std::string> splitString(std::string string, std::string split_by);

	uint getProcessorCount();

}
