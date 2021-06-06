#ifndef __MATRIX_H
#define __MATRIX_H
#include <string>
#include <fstream>
#include <sstream>

namespace nnlib {


	class Matrix {

	public:

		uint width;
		uint height;

		Matrix(uint width, uint height, std::string name = "Matrix");
		Matrix(uint size, std::string name = "Matrix");
		Matrix(std::string path_to_serialized_file);

		void save();
		void save(std::string path_to_serialized_file);
		std::string serialize(uint float_precision = 2, uint float_width = 5) const;
		void deserialize(std::ifstream* serialized);

		float getValue(uint x, uint y) const;
		void setValue(uint x, uint y, float value);

		std::string getName() const;
		void setName(std::string new_name);

		Matrix operator* (const float& n) const;
		Matrix operator/ (const float& n) const;
		Matrix operator+ (const Matrix& v) const;
		Matrix operator- (const Matrix& v) const;
		Matrix operator* (const Matrix& v) const;

		void fillRandom(float min_value = 0, float max_value = 1);
		void fillZero();
		void identity();

		Matrix* clone() const;

		~ Matrix();

	private:
		float ** table;
		std::string name;

		static float ** allocate2DArray(uint width, uint height);

		static void deallocate2DArray(float ** array, uint width, uint height);

		float max_absolute_entry() const;

	};

}

#endif
