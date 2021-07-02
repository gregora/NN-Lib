#include "include/network.h"
#include <iostream>


int main() {

	nnlib::Dense* dense_layer1 = new nnlib::Dense(10, 10, "DLayer1");
	nnlib::Dense* dense_layer2 = new nnlib::Dense(10, 10, "DLayer2");
	nnlib::Dense* dense_layer3 = new nnlib::Dense(10, 10, "DLayer3");
	nnlib::Dense* dense_layer4 = new nnlib::Dense(10, 10, "DLayer4");
	nnlib::Dense* dense_layer5 = new nnlib::Dense(10, 10, "DLayer5");
	nnlib::Dense* dense_layer6 = new nnlib::Dense(10, 10, "DLayer6");

	nnlib::Matrix input(1, 10, "input");
	input.fillRandom();

	nnlib::Network network;

	network.addLayer(dense_layer1);
	network.addLayer(dense_layer2);
	network.addLayer(dense_layer3);
	network.addLayer(dense_layer4);
	network.addLayer(dense_layer5);
	network.addLayer(dense_layer6);

	network.save("test.AI");

	nnlib::Network n2;
	n2.load("test.AI");
	std::cout << n2.serialize();


	return 0;
}
