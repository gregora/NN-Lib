#include "misc.h"
#include <string.h>

namespace nnlib {


	class Matrix {

	public:

		uint width;
		uint height;

		Matrix(uint width, uint height);

		Matrix(uint size);

		void print() const;

		float getValue(uint x, uint y) const;

		void setValue(uint x, uint y, float value);

		Matrix operator* (const float& n);

		Matrix operator/ (const float& n);

		Matrix operator+ (const Matrix& v);

		Matrix operator- (const Matrix& v);

		Matrix operator* (const Matrix& v);

		void fillRandom(float min_value = 0, float max_value = 1);

		String toString();
		Matrix* copy() const;

		~ Matrix();

		~ Matrix();

	private:
		float ** table;

		static float ** allocate2DArray(uint width, uint height);

	};

}
