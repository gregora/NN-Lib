#include "matrix.cpp"

namespace nnlib {

	Matrix(unsigned int height, unsigned int width);

	Matrix(unsigned int size);

	Matrix Matrix::operator* (float n);

	Marix Matrix::operator/ (float n);

	Matrix Matrix::operator+ (Matrix v);

	Matrix Matrix::operator- (Matrix v);

	Matrix Matrix::operator* (Matrix v);

	Matrix Matrix::fillRandom(float min_value = 0, float max_value = 1);

	Matrix Matrix::copy();

	~ Matrix();

}
