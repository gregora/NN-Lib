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

	Matrix Dense::eval(const Matrix* input) {

		Matrix output = (*weights) * (*input) + (*biases);

		//run activation function on output
		for(uint i = 0; i < output.height; i++){
			float value = output.getValue(0, i);
			output.setValue(0, i, activationFunction(value));
		}

		return output;

	}


	std::string Dense::serialize() {
		return getName() + "\n===\n" + weights -> serialize(7) + "\n===\n" + biases -> serialize(7);

	}

	void Dense::deserialize(std::string input) {
		std::vector<std::string> split = splitString(input, "\n===\n");

		setName(split[0]);
		weights -> deserialize(split[1]);
		biases -> deserialize(split[2]);
	}

	Layer* Dense::clone() {
		Layer * copy = new Dense(weights -> width, weights -> height, getName());
		copy -> deserialize(serialize());
		return copy;
	}

	void Dense::setActivationFunction(float (*newActivationFunction)(float)){
		activationFunction = newActivationFunction;
	}


	Dense::~Dense(){
		delete weights;
		delete biases;
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



	Network::Network(){}


	void Network::addLayer(Layer* l){
		layers.push_back(l);
	}

	Matrix Network::eval(const Matrix* input){

		//Next two lines HAVE TO BE TWO SEPARATE LINES. I spent 2 hours on this and i am not doing that again
		Matrix values;
		values = (*input);

		values.setName("values");

		for(uint i = 0; i < layers.size(); i++){
			values = layers[i] -> eval(&values);
		}

		return values;
	}

	std::string Network::serialize(){
		return "";
	}
	void Network::deserialize(std::string input){}

	void Network::save(std::string path){}
	void Network::load(std::string path){}

	// pretty-print; not for exporting
	std::string Network::toString(){
		return "";
	}

	Network* Network::clone(){
		return nullptr;
	}

	Network::~Network(){}

}
