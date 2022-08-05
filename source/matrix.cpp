#include <stdexcept>
#include <random>
#include <iostream>

#include "../include/matrix.h"
#include "../include/misc.h"

namespace nnlib {

	Matrix::Matrix() {
		width = 0;
		height = 0;
		name = "";
	}

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

		// this, because deserialize will try to free() the nonexistent table otherwise
		this -> width = this -> height = 0;

		// read file to std::string
		std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

		this -> deserialize(str);
	}

	void Matrix::save() {
		this->save(this->getName() + ".matrix");
	}

	void Matrix::save(std::string path_to_serialized_file) {
		std::ofstream file(path_to_serialized_file);
		if (!file)
			throw std::invalid_argument("Cannot write to file '"+path_to_serialized_file+"'");
		file << this->serialize();
	}

	float Matrix::getValue(uint x, uint y) const {
		return get(x, y);
	}

	void Matrix::setValue(uint x, uint y, float value) {
		set(x, y, value);
	}

	float Matrix::get(uint x, uint y) const {
		if (x >= width || y >= height || x < 0 || y < 0) {
			std::ostringstream stringStream;
			stringStream << "Fetching value (" << x << "," << y << ")";
			stringStream << " outside of matrix " << getName();
			stringStream << " (h=" << height << " w=" << width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		return table[x][y];
	}

	void Matrix::set(uint x, uint y, float value) {
		if (x >= width || y >= height || x < 0 || y < 0) {
			std::ostringstream stringStream;
			stringStream << "Setting value (" << x << "," << y << ")";
			stringStream << " outside of matrix " << getName();
			stringStream << " (h=" << height << " w=" << width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		table[x][y] = value;
	}

	std::string Matrix::getName() const {
		return this->name;
	}

	void Matrix::setName(std::string new_name) {
		for (uint i = 0; i < new_name.length(); i++)
			//if (isspace(new_name[i]))
			//	throw std::invalid_argument(new_name + " contains whitespace characters!");
		this->name = new_name;
	}

	void Matrix::operator= (const Matrix& matrix) {

		if(matrix.width != width || matrix.height != height){
			deallocate2DArray(table, width, height);

			width = matrix.width;
			height = matrix.height;

			table = allocate2DArray(width, height);
		}


		for(uint i = 0; i < width; i++){
			for(uint j = 0; j < height; j++){
				table[i][j] = matrix.table[i][j];
			}
		}

	}

	Matrix Matrix::operator* (const float& n) const {
		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.table[i][j] = table[i][j] * n;
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
				ret.table[i][j] = table[i][j] / n;
			}
		}

		return ret;
	}

	Matrix Matrix::operator+ (const Matrix& v) const {
		if (width != v.width || height != v.height) {
			std::ostringstream stringStream;
			stringStream << "Adding matrices " << this->getName() << " + " << v.getName();
			stringStream << " of different dimensions ";
			stringStream << " (h=" << this->height << " w=" << this->width << ")";
			stringStream << " and (h=" << v.height << " w=" << v.width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.table[i][j] = table[i][j] + v.table[i][j];
			}
		}

		ret.setName("(" + this->getName() + "+" + v.getName()+")");
		return ret;
	}

	Matrix Matrix::operator- (const Matrix& v) const {
		if (width != v.width || height != v.height) {
			std::ostringstream stringStream;
			stringStream << "Subtracting matrices " << this->getName() << " - " << v.getName();
			stringStream << " of different dimensions ";
			stringStream << " (h=" << this->height << " w=" << this->width << ")";
			stringStream << " and (h=" << v.height << " w=" << v.width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		Matrix ret(width, height);

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				ret.table[i][j] = table[i][j] - v.table[i][j];
			}
		}

		ret.setName("(" + this->getName() + "-" + v.getName()+")");
		return ret;
	}

	Matrix Matrix::operator* (const Matrix& v) const {

		if (width != v.height) {
			std::ostringstream stringStream;
			stringStream << "Multiplying matrices " << this->getName() << " * " << v.getName();
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
					sum += table[k][j] * v.table[i][k];
				}

				ret.table[i][j] = sum;
			}
		}

		ret.setName(this->getName() + "*" + v.getName());
		return ret;
	}

	Matrix Matrix::transpose() const {
		Matrix ret(height, width);

		for(uint i = 0; i < height; i++){
			for(uint j = 0; j < width; j++){
				ret.table[i][j] = table[j][i];
			}
		}

		return ret;
	}

	void Matrix::fillRandom(float min_value, float max_value) {

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				set(i, j, random(min_value, max_value));
			}
		}

	}

	void Matrix::fillZero() {

		for (uint i = 0; i < width; i++) {
			for (uint j = 0; j < height; j++) {
				set(i, j, 0);
			}
		}

	}

	void Matrix::identity() {

		fillZero();
		uint min_dim = width < height? width : height;
		for (uint i = 0; i < min_dim; i++) {
			set(i, i, 1);
		}

	}

	std::string Matrix::serialize(uint float_precision, uint min_float_width) const {

		// get widths for each column
		uint total_width = 0;
		uint* col_width = (uint*)calloc(this->width, sizeof(uint));
		// ISO C++ forbids variable length array, so it has to be malloc()ed
		// (calloc in this case performs faster due to proper uint alignment)

		//uint ten_to_prec = (uint)fast_pow(10, float_precision);

		for (uint i = 0; i < this -> width; i++) {
			// first, calculate the minimum possible width for every column
			float max_absolute_entry = 0;
			char all_entires_positive = 1;
			for (uint j = 0; j < height; j++) {
				float val = get(i, j);
				if (val < 0) {
					all_entires_positive = 0;
					val = -val;
				}
				if (val > max_absolute_entry)
					max_absolute_entry = val;
			}

			col_width[i] = floor(max_absolute_entry);
			col_width[i] = col_width[i] == 0 ? 1 : numlen(col_width[i]);

			// all the required decimal places
			col_width[i] += float_precision;

			// add decimal dot
			if (float_precision != 0)
				col_width[i]++;

			// check if we need to add the sign symbols
			if (!all_entires_positive) {
				col_width[i]++; // sign symbol
			}

			// if more space was requested, grant it
			if (col_width[i] < min_float_width)
				col_width[i] = min_float_width;

			total_width += col_width[i];
		}

		total_width++; // every line ends with "\n"

		const int allocated =
				this->getName().length() // "Matrix"
				+ 9 // " (h= w=)\n"
				+ numlen(width) + numlen(height) // matrix dimensions
				+ height*total_width // matrix data
				+ height*(width - 1) // space between columns
				+ 1; // "\0"
		char* buffer = (char*)malloc(allocated * sizeof(char));
		char* buffer_start = buffer;

		buffer += sprintf(buffer, (this->getName() + " (h=%d w=%d)\n").c_str(), height, width);
		for (uint j = 0; j < height; j++) {
			for (uint i = 0; i < width; i++) {
				buffer += sprintf(buffer,
					"%*.*f",
					col_width[i],
					float_precision,
					get(i, j)
				);
				// spaces between columns, but not on the last column
				if (i+1 < width) buffer += sprintf(buffer, " ");
			}
			buffer += sprintf(buffer, "\n");
		}

		if (buffer - buffer_start + 1 != allocated)
			printf("used %ld, allocated %d\n", buffer - buffer_start, allocated);
		std::string str((const char*)buffer_start);
		free(col_width);
		free(buffer_start);
		return str;
	}

	void Matrix::deserialize(std::string serialized) {
		std::istringstream stream = std::istringstream(serialized);

		deallocate2DArray(this->table, this->width, this->height);

		char c; // useless buffer
		std::string name;
		stream >> name; // "Matrix"
		stream >> c >> c >> c; // "(h="
		stream >> this->height;
		stream >> c >> c; // "w="
		stream >> this->width;
		stream >> c; // ")"

		this->setName(name);
		this->table = allocate2DArray(this->width, this->height);

		for (uint j = 0; j < this->height; j++) {
			for (uint i = 0; i < this->width; i++) {
				float value;
				stream >> value;
				this->set(i, j, value);
			}
		}
	}

	Matrix* Matrix::clone() const{
		Matrix * ret = new Matrix(width, height, this->getName());

		for (uint i = 0; i < width; i++){
			for (uint j = 0; j < height; j++){
				ret -> set(i, j, get(i, j));
			}
		}

		return ret;
	}

	Matrix::~Matrix() {
		deallocate2DArray(table, width, height);
		height = 0;
		width = 0;
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
		if (array == nullptr) return;

		for (uint i = 0; i < width; i++) {
			free(array[i]);
		}

		free(array);
	}


	Matrix dereference(const Matrix* matrix){
		Matrix ret = *matrix;
		ret.table = ret.allocate2DArray(ret.width, ret.height);

		for(unsigned int i = 0; i < ret.width; i++){
			for(unsigned int j = 0; j < ret.height; j++){
				ret.table[i][j] = matrix -> table[i][j];
			}
		}

		return ret;
	}

	void add(Matrix* matrix1, const Matrix* matrix2){
		if (matrix1 -> width != matrix2 -> width || matrix1 -> height != matrix2 -> height) {
			std::ostringstream stringStream;
			stringStream << "Adding matrices " << matrix1->getName() << " - " << matrix2->getName();
			stringStream << " of different dimensions ";
			stringStream << " (h=" << matrix1->height << " w=" << matrix1->width << ")";
			stringStream << " and (h=" << matrix2 -> height << " w=" << matrix2 -> width << ")";
			throw std::invalid_argument(stringStream.str());
		}

		for(uint i = 0; i < matrix1 -> width; i++){
			for(uint j = 0; j < matrix1 -> height; j++){
				matrix1 -> set(i, j, matrix1 -> get(i, j) + matrix2 -> get(i, j));
			}
		}
	}

	void multiply(Matrix* matrix, float n){
		for(uint i = 0; i < matrix -> width; i++){
			for(uint j = 0; j < matrix -> height; j++){
				matrix -> set(i, j, matrix -> get(i, j) * n);
			}
		}
	}


}
