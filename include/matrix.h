#include "misc.h"

namespace nnlib {


	class Matrix {

	public:

		uint width;
		uint height;

		Matrix(uint width, uint height);
		Matrix(uint size);

		float getValue(uint x, uint y) const;
		void setValue(uint x, uint y, float value);

		Matrix operator* (const float& n) const;
		Matrix operator/ (const float& n) const;
		Matrix operator+ (const Matrix& v) const;
		Matrix operator- (const Matrix& v) const;
		Matrix operator* (const Matrix& v) const;

		void fillRandom(float min_value = 0, float max_value = 1);
		void fillZero();
		void identity();

		std::string serialize(uint float_width = 5, uint float_precision = 2) const;
		void deserialize();

		void print(uint float_width = 5, uint float_precision = 2) const;

		Matrix* clone() const;

		~ Matrix();

	private:
		float ** table;

		static float ** allocate2DArray(uint width, uint height);

		static void deallocate2DArray(float ** array, uint width, uint height);

		float max() const; // get maximum absolute value in the matrix

	};

}
