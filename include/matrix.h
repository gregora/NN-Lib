#include "misc.h"

namespace nnlib {


	class Matrix {

	public:

		unsigned int width;
		unsigned int height;


		Matrix(unsigned int height, unsigned int width);

		Matrix(unsigned int size);

		Matrix operator* (float n);

		Marix operator/ (float n);

		Matrix operator+ (Matrix v);

		Matrix operator- (Matrix v);

		Matrix operator* (Matrix v);

		Matrix fillRandom(float min_value = 0, float max_value = 1);

		Matrix copy();

		~ Matrix();

	}

}
