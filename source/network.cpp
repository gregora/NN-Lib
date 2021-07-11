#include "../include/network.h"

namespace nnlib {

	Dense::Dense(uint input, uint output, std::string name) {

		type = "Dense";

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
		return getName() + "\n===\n" + weights -> serialize(7) + "===\n" + biases -> serialize(7);

	}

	void Dense::deserialize(std::string input) {
		std::vector<std::string> split = splitString(input, "\n===\n");
		if (split.size() != 3)
			throw std::invalid_argument("Attempting to deserialize an invalid Dense layer format");
		setName(split[0]);
		weights -> deserialize(split[1] + "\n");
		biases -> deserialize(split[2]);
	}

	Layer* Dense::clone() {
		Layer * copy = new Dense(weights -> width, weights -> height, getName());
		copy -> deserialize(serialize());
		return copy;
	}

	void Dense::setActivationFunction(float (*newActivationFunction)(float)) {
		activationFunction = newActivationFunction;
	}


	Dense::~Dense() {
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



	Network::Network(std::string name) {
		this->setName(name);
	}


	void Network::addLayer(Layer* l) {
		layers.push_back(l);
	}

	Matrix Network::eval(const Matrix* input) {

		//Next two lines HAVE TO BE TWO SEPARATE LINES. I spent 2 hours on this and i am not doing that again
		Matrix values;
		values = (*input);

		values.setName(this->getName() + "_evaluated_on_" + input->getName());

		for(uint i = 0; i < layers.size(); i++) {
			values = layers[i] -> eval(&values);
		}

		return values;
	}

	std::string Network::serialize() {
		std::string ret = this->getName() + "\n";

		for(uint i = 0; i < layers.size(); i++) {
			ret += "====================\nLayer: ";
			ret += layers[i] -> type + ";\n";
			ret += layers[i] -> serialize();
		}

		return ret;
	}
	void Network::deserialize(std::string input) {

		for(uint i = 0; i < layers.size(); i++) {
			delete layers[i];
		}

		layers.clear();

		std::vector<std::string> splitLayers = splitString(input, "====================\nLayer: ");
		if (splitLayers.size() < 2)
			throw std::invalid_argument("Deserializing a Network without layers");

		setName(splitLayers[0]);

		for (uint i = 1; i < splitLayers.size(); i++) {

			std::vector<std::string> split2 = splitString(splitLayers[i], ";\n");
			if (split2.size() < 1)
				throw std::invalid_argument("Deserializing a Network with an empty layer (#"+std::to_string(i)+")");

			Layer* layer;
			if (split2[0] == "Dense") {
				layer = new Dense(1, 1);
				layer -> deserialize(split2[1]);
			}
			else {
				throw std::invalid_argument("Invalid layer type: " + split2[0] + " in " + input);
			}

			layers.push_back(layer);

		}
	}

	void Network::save() {
		this->save(this->getName());
	}

	void Network::save(std::string path) {
		std::ofstream file(path);
		file << serialize();
		file.close();
	}

	void Network::load(std::string path) {
		std::ifstream file;
		file.open(path);

		std::stringstream buffer;
		buffer << file.rdbuf(); //read the file
		std::string string = buffer.str(); //str holds the content of the file

		deserialize(string);
	}

	// pretty-print; not for exporting
	void Network::print() {
		printf("Network pretty print:\n");
		for(uint i = 0; i < layers.size(); i++){
			printf(" %d. %s (%s)\n", i, layers[i] -> getName().c_str(), layers[i] -> type.c_str());
		}
	}

	Network* Network::clone() {
		Network* ret = new Network;
		ret -> deserialize(serialize());

		return ret;
	}

	std::string Network::getName() const {
		return this->name;
	}

	void Network::setName(std::string new_name) {
		this->name = new_name;
	}

	Network::~Network(){
		for(uint i = 0; i < layers.size(); i++){
			delete layers[i];
		}
	}

}
