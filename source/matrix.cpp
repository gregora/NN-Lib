#include "../include/matrix.h"

namespace nnlib {

	Matrix::Matrix(uint width, uint height) {
		this -> height = height;
		this -> width = width;

		table = allocate2DArray(width, height);
	}

	Matrix::Matrix(uint size) {
		this -> height = size;
		this -> width = size;

		table = allocate2DArray(width, height);
	}

	void Matrix::print() const {

		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				printf("%f ", table[i][j]);
			}
			printf("\n");
		}

	}

	float Matrix::getValue(uint x, uint y) const {

		if (x >= width || y >= height)
			throw std::invalid_argument("Fetching value outside of matrix");

		return table[x][y];
	}

	void Matrix::setValue(uint x, uint y, float value) {

		if (x >= width || y >= height)
			throw std::invalid_argument("Setting value outside of matrix");

		table[x][y] = value;
	}

	Matrix Matrix::operator* (const float& n) {
		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) * n);
			}
		}

		return ret;
	}

	Matrix Matrix::operator/ (const float& n) {
		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) / n);
			}
		}

		return ret;
	}

	Matrix Matrix::operator+ (const Matrix& v) {
		if(width != v.width || height != v.height)
			throw std::invalid_argument("Adding up matrices of different sizes");

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) + v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator- (const Matrix& v) {
		if(width != v.width || height != v.height)
			throw std::invalid_argument("Subtracting matrices of different sizes");

		Matrix ret(width, height);

		for(uint i = 0; i < width; i++){
			for(uint j = 0; j < height; j++){
				ret.setValue(i, j, getValue(i, j) - v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator* (const Matrix& v) {
		if(width != v.height)
			throw std::invalid_argument("Multiplying matrices of invalid sizes");

		Matrix ret(v.width, height);

		for (int i = 0; i < v.width; i++) {
			for (int j = 0; j < height; j++) {
				float sum = 0;

				for (int k = 0; k < width; k++) {
					sum += getValue(k, j) * v.getValue(i, k);
				}

				ret.setValue(i, j, sum);
			}
		}

		return ret;
	}

	void Matrix::fillRandom(float min_value, float max_value) {

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				setValue(i, j, min_value + (max_value - min_value)*random());
			}
		}

	}

	char* Matrix::toString(uint float_width, uint float_precision) {
		// +2: one for sign and one for the decimal separator
		const int float_len = float_width + float_precision + 2;

		char* buffer = (char*)malloc((
				2 // "[\n"
				+ height*(width*(float_len + 1)) // "%f " for all elements
				+ height*1 // plus one newline per height
				+ 3 // "]\n\0"
			) * sizeof(char)
		);
		char* buffer_start = buffer;

		buffer += sprintf(buffer, "[\n");
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				buffer += sprintf(buffer,
					"% *.*f ",
					float_width,
					float_precision,
					table[i][j]
				);
			}
			buffer += sprintf(buffer, "\n");
		}
		buffer += sprintf(buffer, "]\n");
		return buffer_start;
	}

	Matrix* Matrix::copy() const{
		Matrix * ret = new Matrix(width, height);

		for(int i = 0; i < width; i++){
			for(int j = 0; j < height; j++){
				ret -> setValue(i, j, getValue(i, j));
			}
		}

		return ret;
	}

	Matrix::~Matrix(){
		for (int i = 0; i < width; i++) {
			free(table[i]);
		}
		free(table);
	}



	float ** Matrix::allocate2DArray(uint width, uint height){

		float ** ret = (float**) calloc(width, sizeof(float *));
		for (int i = 0; i < width; i++) {
			ret[i] = (float*) calloc(height, sizeof(float));
		}

		return ret;

	}

}
