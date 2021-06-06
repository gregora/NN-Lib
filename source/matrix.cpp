#include <stdexcept>
#include <random>

#include "../include/matrix.h"
#include "../include/misc.h"

namespace nnlib {

	Matrix::Matrix(uint width, uint height, std::string name) {
		this -> height = height;
		this -> width = width;
		this -> setName(name);

		table = allocate2DArray(width, height);
	}

	Matrix::Matrix(uint size, std::string name) {
		this -> height = size;
		this -> width = size;
		this -> setName(name);

		table = allocate2DArray(width, height);
	}

	Matrix::Matrix(std::string path_to_serialized_file) {
		std::ifstream file(path_to_serialized_file);
		if (!file)
			throw std::invalid_argument("Cannot read file '"+path_to_serialized_file+"'");
		this -> width = this -> height = 0;
		this -> deserialize(&file);
	}

	void Matrix::save() {
		this->save(this->name + ".matrix");
	}

	void Matrix::save(std::string path_to_serialized_file) {
		std::ofstream file(path_to_serialized_file);
		if (!file)
			throw std::invalid_argument("Cannot write to file '"+path_to_serialized_file+"'");
		file << this->serialize();
	}

	float Matrix::getValue(uint x, uint y) const {

		if (x >= width || y >= height || x < 0 || y < 0) {
			std::ostringstream stringStream;
			stringStream << "Fetching value (" << x << "," << y << ")";
			stringStream << " outside of matrix " << name;
			stringStream << " (h=" << height << " w=" << width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		return table[x][y];
	}

	void Matrix::setValue(uint x, uint y, float value) {

		if (x >= width || y >= height || x < 0 || y < 0) {
			std::ostringstream stringStream;
			stringStream << "Setting value (" << x << "," << y << ")";
			stringStream << " outside of matrix " << name;
			stringStream << " (h=" << height << " w=" << width << ")";
			throw std::invalid_argument(stringStream.str());
		};

		table[x][y] = value;
	}

	std::string Matrix::getName() const {
		return this->name;
	}

	void Matrix::setName(std::string new_name) {
		for (uint i = 0; i < new_name.length(); i++)
			if (isspace(new_name[i]))
				throw std::invalid_argument(new_name + " contains whitespace characters!");
		this->name = new_name;
	}

	Matrix Matrix::operator* (const float& n) const {
		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) * n);
			}
		}

		return ret;
	}

	Matrix Matrix::operator/ (const float& n) const {
		if (n == 0)
			throw std::invalid_argument("Dividing " + this->getName() + " by 0!");

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) / n);
			}
		}

		return ret;
	}

	Matrix Matrix::operator+ (const Matrix& v) const {
		if (width != v.width || height != v.height) {
			std::ostringstream stringStream;
			stringStream << "Adding matrices " << this->name << " + " << v.name;
			stringStream << " of different dimensions ";
			stringStream << " (h=" << this->height << " w=" << this->width << ")";
			stringStream << " and (h=" << v.height << " w=" << v.width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) + v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator- (const Matrix& v) const {
		if (width != v.width || height != v.height) {
			std::ostringstream stringStream;
			stringStream << "Subtracting matrices " << this->name << " - " << v.name;
			stringStream << " of different dimensions ";
			stringStream << " (h=" << this->height << " w=" << this->width << ")";
			stringStream << " and (h=" << v.height << " w=" << v.width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.setValue(i, j, getValue(i, j) - v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator* (const Matrix& v) const {
		if (width != v.height) {
			std::ostringstream stringStream;
			stringStream << "Multiplying matrices " << this->name << " * " << v.name;
			stringStream << " of invalid dimensions ";
			stringStream << " (h=" << this->height << " w=" << this->width << ")";
			stringStream << " and (h=" << v.height << " w=" << v.width << ")";
			throw std::invalid_argument(stringStream.str());
		}

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

	std::string Matrix::serialize(uint float_precision, uint float_width) const {
		// TODO: only expand the columns as needed

		// first, get the length of the maximum entry
		uint max_length = floor(max_absolute_entry());
		if (max_length != 0)
			max_length = numlen(max_length);
		else max_length = 1;

		// assure float_width fits the maximum entry
		if (float_width < float_precision + 2 + max_length)
			float_width = float_precision + 2 + max_length;
		// +2 places account for the decimal dot and sign symbol

		const int allocated =
				15 // "Matrix (h= w=)\n"
				+ numlen(width) + numlen(height)
				+ height*(width*(float_width + 1)) // "%f " for all elements
				+ height*1 // plus one newline per height
				+ 1; // "\0"
		char* buffer = (char*)malloc(allocated * sizeof(char));
		char* buffer_start = buffer;

		buffer += sprintf(buffer, (this->name + " (h=%d w=%d)\n").c_str(), height, width);
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
		//printf("used %d, allocated %d\n", buffer - buffer_start, allocated);
		std::string str((const char*)buffer_start);
		free(buffer_start);
		return str;
	}

	void Matrix::deserialize(std::ifstream* serialized) {
		deallocate2DArray(this->table, this->width, this->height);

		char c; // useless buffer
		*serialized >> this->name; // "Matrix"
		*serialized >> c >> c >> c; // "(h="
		*serialized >> this->height;
		*serialized >> c >> c; // "w="
		*serialized >> this->width;
		*serialized >> c; // ")"

		this->table = allocate2DArray(this->width, this->height);

		for (uint j = 0; j < this->height; j++) {
			for (uint i = 0; i < this->width; i++) {
				float value;
				*serialized >> value;
				this->setValue(i, j, value);
			}
		}
	}

	// needed to calculate space for allocation in toBuffer, toString and print
	float Matrix::max_absolute_entry() const {
		// get maximum absolute value in the matrix
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

	Matrix* Matrix::clone() const{
		Matrix * ret = new Matrix(width, height, this->name);

		for (uint i = 0; i < width; i++){
			for (uint j = 0; j < height; j++){
				ret -> setValue(i, j, getValue(i, j));
			}
		}

		return ret;
	}

	Matrix::~Matrix(){
		deallocate2DArray(table, width, height);
	}



	float ** Matrix::allocate2DArray(uint width, uint height) {

		float ** ret = (float**) calloc(width, sizeof(float *));
		for (uint i = 0; i < width; i++) {
			ret[i] = (float*) calloc(height, sizeof(float));
		}

		return ret;

	}

	void Matrix::deallocate2DArray(float ** array, uint width, uint height) {
		if (width == 0 || height == 0) return;

		for (uint i = 0; i < width; i++) {
			free(array[i]);
		}
		free(array);
	}
}
