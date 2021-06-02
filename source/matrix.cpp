#include "../include/matrix.h"

namespace nnlib {

	Matrix::Matrix(uint width, uint height){
		this -> height = height;
		this -> width = width;

		table = (float**) calloc(width, sizeof(float *));
		for (int i = 0; i < width; i++) {
			table[i] = (float*) calloc(height, sizeof(float));
		}
	}

	Matrix::Matrix(uint size){
		this -> height = size;
		this -> width = size;

		table = (float**) calloc(width, sizeof(float *));
		for (int i = 0; i < width; i++) {
			table[i] = (float*) calloc(height, sizeof(float));
		}
	}

	void Matrix::print(){

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				printf("%f ", table[i][j]);
			}
			printf("\n");
		}

	}

	float Matrix::getValue(uint x, uint y){

		if(x >= width || y >= height)
			throw std::invalid_argument("Fetching value outside of matrix");

		return table[x][y];
	}

	void Matrix::setValue(uint x, uint y, float value){

		if(x >= width || y >= height)
			throw std::invalid_argument("Setting value outside of matrix");

		table[x][y] = value;
	}

	Matrix Matrix::operator* (float n){
		Matrix ret(width, height);

		for(uint i = 0; i < width; i++){
			for(uint j = 0; j < height; j++){
				ret.setValue(i, j, getValue(i, j) * n);
			}
		}

		return ret;
	}

	Matrix Matrix::operator/ (float n){
		Matrix ret(width, height);

		for(uint i = 0; i < width; i++){
			for(uint j = 0; j < height; j++){
				ret.setValue(i, j, getValue(i, j) / n);
			}
		}

		return ret;
	}

	Matrix Matrix::operator+ (Matrix v){
		if(width != v.width || height != v.height)
			throw std::invalid_argument("Adding up matrices of different sizes");

		Matrix ret(width, height);

		for(uint i = 0; i < width; i++){
			for(uint j = 0; j < height; j++){
				ret.setValue(i, j, getValue(i, j) + v.getValue(i, j));
			}
		}

		return ret;
	}

	Matrix Matrix::operator- (Matrix v){
		if(width != v.width || height != v.height)
			throw std::invalid_argument("Subtracting matrices of different sizes");

		Matrix ret(width, height);

		ret = (*this) + (v*-1);

		return ret;
	}

	Matrix Matrix::operator* (Matrix v){
		if(width != v.height)
			throw std::invalid_argument("Multiplying matrices of invalid sizes");

		Matrix ret(v.width, height);

		for(int i = 0; i < v.width; i++){
			for(int j = 0; j < height; j++){
				float sum = 0;

				for(int k = 0; k < width; k++){
					sum += getValue(k, j) * v.getValue(i, k);
				}

				ret.setValue(i, j, sum);
			}
		}

		return ret;
	}

	Matrix Matrix::fillRandom(float min_value, float max_value){

		for(int i = 0; i < width; i++){
			for(int j = 0; j < height; j++){
				setValue(i, j, min_value + (max_value - min_value)*random());
			}
		}

	}

	String toString() {
		std::stringstream buffer;
		sprintf(buffer, "[\n");
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				sprintf(buffer, "%f ", table[i][j]);
			}
			sprintf(buffer, "\n");
		}
		sprintf(buffer, "]\n");
		return buffer.str();
	}

	//Matrix Matrix::copy();

	~ Matrix() {
		for (int i = 0; i < width; i++) {
			free(table[i]);
		}
		free(table);
	}

}
