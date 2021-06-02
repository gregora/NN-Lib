#include "include/matrix.h"
#include <iostream>

int main() {

	nnlib::Matrix a(5, 5);
	nnlib::Matrix b(5, 5);
	a.fillRandom();
	a.setValue(0, 0, 3);
	a.setValue(0, 3, 8);
	a.setValue(0, 3, -8);
	a.setValue(1, 3, -0.22333);
	a.setValue(3, 3, 10000);
	a.setValue(2, 3, 0);
	b.identity();

	a*b;

	nnlib::Matrix * c = a.copy();

	std::cout << b.toString();
	a.print();
	//c -> print();

	printf("QED\n");
	return 0;
}
