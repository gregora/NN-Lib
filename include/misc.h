#include "matrix.h"
//is this double declaration of <vector> and <string>?
#include <vector>
#include <string>
#include <thread>
#include <math.h>
#include <string.h>

namespace nnlib {

	// uniformly distributed random float between 0 and 1
	float random();
	float random(float min, float max);
	int randomInt(int min, int max);

	double fast_pow(double base, uint exponent);

	// number of digits x takes up when printed (numlen(0) = 1)
	int numlen(int x);


	// activation functions (function evaluated at vector)
	Matrix linear(const Matrix& x);
	Matrix sigmoid(const Matrix& x);
	Matrix fast_sigmoid(const Matrix& x);
	Matrix relu(const Matrix& x);
	Matrix atan(const Matrix& x);
	Matrix tanh(const Matrix& x);
	Matrix softmax(const Matrix& x);
	//activation function derivatives (Jacobian matrices evaluated at vector)
	Matrix dlinear(const Matrix& x);
	Matrix dsigmoid(const Matrix& x);
	Matrix dfast_sigmoid(const Matrix& x);
	Matrix drelu(const Matrix& x);
	Matrix datan(const Matrix& x);
	Matrix dtanh(const Matrix& x);
	Matrix dsoftmax(const Matrix& x);

	float sparseCategoricalCrossentropy(Matrix* predicted, uint truth);
	float categoricalCrossentropy(Matrix* predicted, Matrix* truth);
	float binaryCrossentropy(Matrix* predicted, Matrix* truth);
	float meanSquaredError(const Matrix* predicted, const Matrix* truth);


	std::vector<std::string> splitString(std::string string, std::string split_by);

	uint getProcessorCount();

}
