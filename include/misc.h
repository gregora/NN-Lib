#include <random>
#include <stdexcept>

namespace nnlib {

	// uniformly distributed random float between 0 and 1
	float random();
	float random(float min, float max);

	float sigmoid(float x);

	float sparseCategoricalCrossentropy(Matrix* predicted, uint truth);

	float categoricalCrossentropy(Matrix* predicted, Matrix* truth);

	float binaryCrossentropy(Matrix* predicted, Matrix* truth);

	float meanSquaredError(Matrix* predicted, Matrix* truth);
}
