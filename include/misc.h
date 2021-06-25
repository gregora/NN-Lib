#include "matrix.h"

namespace nnlib {

	// uniformly distributed random float between 0 and 1
	float random();
	float random(float min, float max);

	double fast_pow(double base, uint exponent);

	// number of digits x takes up when printed (numlen(0) = 1)
	int numlen(int x);

	float sigmoid(float x);

	float fast_sigmoid(float x);

	float sparseCategoricalCrossentropy(Matrix* predicted, uint truth);

	float categoricalCrossentropy(Matrix* predicted, Matrix* truth);

	float binaryCrossentropy(Matrix* predicted, Matrix* truth);

	float meanSquaredError(Matrix* predicted, Matrix* truth);
}
