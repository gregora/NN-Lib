#include "include/matrix.h"

int main(){

	nnlib::Matrix a(5, 5);
	nnlib::Matrix b(5, 5);
	a.fillRandom();
	a.setValue(0, 0, 3);
	a.setValue(0, 3, 8);
	a.setValue(0, 3, -8);
	a.setValue(1, 3, -0.22333);
	b.setValue(1, 1, 1);

	a*b;

	nnlib::Matrix * c = a.copy();

	printf("%s\n", a.toString());
	//c -> print();

	printf("END OF THE PROG.\n");

}
