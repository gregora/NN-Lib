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
		if (width != v.width || height != v.height)
			throw std::invalid_argument("Adding matrices of different sizes");

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) + v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator- (const Matrix& v) {
		if (width != v.width || height != v.height)
			throw std::invalid_argument("Subtracting matrices of different sizes");

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) - v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator* (const Matrix& v) {
		if (width != v.height)
			throw std::invalid_argument("Multiplying matrices of invalid sizes");

		Matrix ret(v.width, height);

		for (uint i = 0; i < v.width; i++) {
			for (uint j = 0; j < height; j++) {
				float sum = 0;

				for (uint k = 0; k < width; k++) {
					sum += getValue(k, j) * v.getValue(i, k);
				}

				ret.setValue(i, j, sum);
			}
		}

		return ret;
	}

	void Matrix::fillRandom(float min_value, float max_value) {

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				setValue(i, j, random(min_value, max_value));
			}
		}

	}

	void Matrix::fillZero() {

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				setValue(i, j, 0);
			}
		}

	}

	void Matrix::identity() {

		fillZero();
		uint min_dim = width < height? width : height;
		for (uint i = 0; i < min_dim; i++) {
			setValue(i, i, 1);
		}

	}

	void Matrix::print(uint float_width, uint float_precision) const {
		char* buff = this->toBuffer(float_width, float_precision);
		printf("%s", buff);
		free(buff);
		// alternatively: cout << toString();
	}

	std::string Matrix::toString(uint float_width, uint float_precision) const {
		char* buffer = toBuffer(float_width, float_precision);
		std::string str((const char*)buffer);
		free(buffer);
		return str;
	}

	char* Matrix::toBuffer(uint float_width, uint float_precision) const {
		// TODO: only expand the columns as needed

		// first, get the length of the maximum entry
		uint max_length = floor(max());
		if (max_length != 0)
			max_length = floor(log10(max_length)) + 1;
		else max_length = 1;

		// assure float_width fits the maximum entry
		if (float_width < float_precision + 2 + max_length)
			float_width = float_precision + 2 + max_length;
		// +2 places account for the decimal dot and sign symbol

		const int allocated =
				2 // "[\n"
				+ height*(width*(float_width + 1)) // "%f " for all elements
				+ height*1 // plus one newline per height
				+ 3; // "]\n\0"
		char* buffer = (char*)malloc(allocated * sizeof(char));
		char* buffer_start = buffer;

		buffer += sprintf(buffer, "[\n");
		for (uint j = 0; j < height; j++) {
			for (uint i = 0; i < width; i++) {
				buffer += sprintf(buffer,
					"% *.*f ",
					float_width,
					float_precision,
					getValue(i, j)
				);
			}
			buffer += sprintf(buffer, "\n");
		}
		buffer += sprintf(buffer, "]\n");
		//printf("used %d, allocated %d\n", buffer - buffer_start, allocated);
		return buffer_start;
	}

	// needed to calculate space for allocation in toBuffer, toString and print
	float Matrix::max() const {
		float m = 0;
		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				float val = getValue(i, j);
				val = val > 0? val : -val;
				if (val > m)
					m = val;
			}
		}
		return m;
	}

	Matrix* Matrix::copy() const{
		Matrix * ret = new Matrix(width, height);

		for (uint i = 0; i < width; i++){
			for (uint j = 0; j < height; j++){
				ret -> setValue(i, j, getValue(i, j));
			}
		}

		return ret;
	}

	Matrix::~Matrix(){
		for (uint i = 0; i < width; i++) {
			free(table[i]);
		}
		free(table);
	}



	float ** Matrix::allocate2DArray(uint width, uint height){

		float ** ret = (float**) calloc(width, sizeof(float *));
		for (uint i = 0; i < width; i++) {
			ret[i] = (float*) calloc(height, sizeof(float));
		}

		return ret;

	}

}
