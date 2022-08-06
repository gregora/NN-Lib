#ifndef __MATRIX_H
#define __MATRIX_H
#include <string>
#include <fstream>
#include <sstream>

namespace nnlib {

	class Matrix {

	public:
		float ** table = nullptr;

		uint width;
		uint height;

		Matrix();
		Matrix(uint width, uint height, std::string name = "Matrix");
		Matrix(uint size, std::string name = "Matrix");
		Matrix(std::string path_to_serialized_file);

		void save();
		void save(std::string path_to_serialized_file);
		std::string serialize(uint float_precision = 2, uint min_float_width = 0) const;
		void deserialize(std::string serialized);

		float getValue(uint x, uint y) const;
		void setValue(uint x, uint y, float value);

		float get(uint x, uint y) const;
		void set(uint x, uint y, float value);

		std::string getName() const;
		void setName(std::string new_name);

		void operator= (const Matrix& matrix);
		Matrix operator* (const float& n) const;
		Matrix operator/ (const float& n) const;
		Matrix operator+ (const Matrix& v) const;
		Matrix operator- (const Matrix& v) const;
		Matrix operator* (const Matrix& v) const;

		Matrix transpose() const;

		void fillRandom(float min_value = 0, float max_value = 1);
		void fillZero();
		void identity();

		Matrix* clone() const;

		~Matrix();

		static float ** allocate2DArray(uint width, uint height);
		static void deallocate2DArray(float ** array, uint width, uint height);

	private:
		std::string name;


	};

	Matrix dereference(const Matrix* matrix);

	void add(Matrix* matrix1, const Matrix* matrix2);
	void multiply(Matrix* matrix, float n);

}

#endif
