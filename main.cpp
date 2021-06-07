#include "include/network.h"
#include <iostream>

int main() {

	nnlib::Matrix a(5, 5, "MatA");
	nnlib::Matrix b(5, 5, "MatB");
	a.fillRandom();
	a.setValue(0, 0, 3);
	a.setValue(0, 3, 9.999);
	a.setValue(1, 3, -0.22333);
	a.setValue(3, 3, 10000);
	a.setValue(2, 3, 0);
	b.identity();

	a*b;

	nnlib::Matrix * c = a.clone();
	c->setName("C_(clone-of-A)");

	std::cout << b.serialize();

	printf("matrix c: %s", c -> serialize().c_str());

	/*
	nnlib::Network model = nnlib::Network();
	model.addLayer(nnlib::Dense(input_size));
	model.addLayer(nnlib::Dense(hidden_size));
	model.addLayer(nnlib::Dense(hidden_size));
	model.addLayer(nnlib::Dense(output_size));

	model.load_weights(checkpoint_dir);
	model.optimizer = nnlib::Adam(learning_rate);

	int i = 0;
	while (true) {
		float* x_batch, y_batch = ...;
		uinx x_count, y_count;
		float y_hat = model.eval(x, x_count);
		Matrix gradient = model.backpropagate(y, y_hat);
		float loss = sparseCategoricalCrossentropy(y, y_hat, y_count);
		printf("%d %7.3f\n", i, loss);
		i++;
	}
	*/

	nnlib::Matrix("identity5.matrix");

	nnlib::Matrix horse(4, 8, "Horse");
	horse.fillRandom(0, 10);
	horse.save();

	nnlib::Matrix bluaberry_pie("Horse.matrix");
	std::cout << bluaberry_pie.serialize(4, 7);

	printf("QED\n");
	return 0;
}
