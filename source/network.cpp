#include "../include/network.h"

namespace nnlib {

	Dense::Dense(uint input, uint output, std::string name) {
		weights = new Matrix(input, output);
		weights -> fillRandom();

		biases = new Matrix(1, output);
		biases -> fillRandom();

		setName(name);
		weights -> setName(name + "_weights");
		biases -> setName(name + "_biases");
	}

	Matrix Dense::eval(Matrix input){

		return (*weights) * input + (*biases);

	}


	std::string Dense::serialize() {
		return NULL;

	}
	void Dense::deserialize(std::string input) {

	}

	Layer* Dense::clone() {
		return NULL;
	}


	std::string Layer::getName() const {
		return this->name;
	}

	void Layer::setName(std::string new_name) {
		for (uint i = 0; i < new_name.length(); i++)
			if (isspace(new_name[i]))
				throw std::invalid_argument(new_name + " contains whitespace characters!");
		this->name = new_name;
	}
}
