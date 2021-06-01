#include "include/matrix.h"

int main(){

	nnlib::Matrix a(5, 5);
	nnlib::Matrix b(5, 5);
	a.setValue(0, 0, 3);
	a.setValue(0, 3, 8);
	b.setValue(1, 1, 1);
	a = a - b;
	(a).print();

}
