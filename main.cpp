#include "include/network.h"
#include <iostream>


float func(float a){
	return a;
}

int main() {

	nnlib::Dense dense_layer(10, 1, "DLayer");

	nnlib::Matrix input(1, 10, "input");
	input.fillRandom();

	//std::cout << input.serialize() << "\n";

	//std::cout << (dense_layer.eval(&input).serialize()) << "\n";

	std::cout << dense_layer.serialize();

	std::cout << "\n\n";

	nnlib::Dense dense_layer2(10, 1, "DLayer2");
	std::cout << dense_layer2.serialize();

	std::cout << "\n\n";

	dense_layer2.deserialize(dense_layer.serialize());
	std::cout << dense_layer2.serialize();


	return 0;
}
