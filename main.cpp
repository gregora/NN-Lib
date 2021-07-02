#include "include/network.h"
#include <iostream>


float func(float a){
	return a;
}

int main() {

	nnlib::Dense dense_layer1(10, 10, "DLayer1");
	nnlib::Dense dense_layer2(10, 10, "DLayer2");
	nnlib::Dense dense_layer3(10, 10, "DLayer3");
	nnlib::Dense dense_layer4(10, 10, "DLayer4");
	nnlib::Dense dense_layer5(10, 10, "DLayer5");
	nnlib::Dense dense_layer6(10, 10, "DLayer6");

	nnlib::Matrix input(1, 10, "input");
	input.fillRandom();

	nnlib::Network network;

	network.addLayer(&dense_layer1);
	network.addLayer(&dense_layer2);
	network.addLayer(&dense_layer3);
	network.addLayer(&dense_layer4);
	network.addLayer(&dense_layer5);
	network.addLayer(&dense_layer6);

	nnlib::Matrix a = network.eval(&input);
	a.setName("a");

	std::cout << input.serialize() << std::endl;

	//std::cout << dense_layer1.serialize() << std::endl;
	//network.eval(&input);

	//std::cout << a.serialize() << std::endl;

	return 0;
}
