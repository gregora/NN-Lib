#include "include/network.h"
#include <iostream>

int main() {

	nnlib::Dense dense_layer(10, 1);

	nnlib::Matrix input(1, 10);
	input.fillRandom();

	std::cout << input.serialize() << "\n";

	std::cout << (dense_layer.eval(input).serialize()) << "\n";

}
