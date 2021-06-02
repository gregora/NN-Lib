#include <random>
#include <stdexcept>

namespace nnlib {

	// uniformly distributed random float between 0 and 1
	float random();
	float random(float min, float max);

	float sigmoid(float x);

	float sparseCategoricalCrossentropy(float* predicted, float truth);

	float categoricalCrossentropy(float* predicted, float* truth);

	float binaryCrossentropy(float* predicted, float* truth);

	float meanSquaredError(float* predicted, float* truth);
}
