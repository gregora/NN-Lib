#include "misc.h"

namespace nnlib {


	class Matrix {

	public:

		uint width;
		uint height;

		Matrix(uint width, uint height);

		Matrix(uint size);

		void print();

		float getValue(uint x, uint y);

		void setValue(uint x, uint y, float value);

		Matrix operator* (float n);

		Matrix operator/ (float n);

		Matrix operator+ (Matrix v);

		Matrix operator- (Matrix v);

		Matrix operator* (Matrix v);

		Matrix fillRandom(float min_value = 0, float max_value = 1);

		/*Matrix copy();

		~ Matrix();*/

	private:
		float ** table;

	};

}
