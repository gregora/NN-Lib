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

		Matrix operator* (const float& n);

		Matrix operator/ (const float& n);

		Matrix operator+ (const Matrix& v);

		Matrix operator- (const Matrix& v);

		Matrix operator* (const Matrix& v);

		void fillRandom(float min_value = 0, float max_value = 1);

		// if float_width is too small, the smallest possible value will be used
		void           print(uint float_width = 5, uint float_precision = 2) const;
		std::string toString(uint float_width = 5, uint float_precision = 2) const;
		char*       toBuffer(uint float_width = 5, uint float_precision = 2) const;

		Matrix* copy() const;

		~ Matrix();

	private:
		float ** table;

		static float ** allocate2DArray(uint width, uint height);

		float max() const; // get maximum absolute value in the matrix

	};

}
