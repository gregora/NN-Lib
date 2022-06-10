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

		Matrix output = dereference(weights) * dereference(input) + dereference(biases);

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


	void Dense::randomize(float min, float max){
		weights -> fillRandom(min, max);
		biases -> fillRandom(min, max);
	}

	void Dense::mutate(float min, float max){
		float rand = random();
		int weights_num = weights->width * weights->height;
		int biases_num = biases->width * biases->height;

		if(rand < (float)weights_num / (weights_num + biases_num)){
			//change random weight
			int x = randomInt(0, weights->width - 1);
			int y = randomInt(0, weights->height - 1);
			weights -> setValue(x, y, random(min, max));
		}else{
			//change random bias
			int x = randomInt(0, biases->width - 1);
			int y = randomInt(0, biases->height - 1);
			biases -> setValue(x, y, random(min, max));
		}
	}

	Dense* Dense::crossover(const Dense* b) const{
		if(b -> inputSize() == inputSize() && b -> outputSize() == outputSize()){
			Dense* ret = new Dense(inputSize(), outputSize());

			for(int j = 0; j < outputSize(); j++){
				for(int i = 0; i < inputSize(); i++){
					if(random() > 0.5){
						ret -> weights -> setValue(i, j, weights -> getValue(i, j));
					}else{
						ret -> weights -> setValue(i, j, b -> weights -> getValue(i, j));
					}
				}

				if(random() > 0.5){
					ret -> biases -> setValue(0, j, biases -> getValue(0, j));
				}else{
					ret -> biases -> setValue(0, j, b -> biases -> getValue(0, j));
				}

			}

			return ret;

		}else{
			throw std::invalid_argument("Combining layers of different sizes!");
		}

	}


	Dense* Dense::crossover_avg(const Dense* b) const{
		if(b -> inputSize() == inputSize() && b -> outputSize() == outputSize()){
			Dense* ret = new Dense(inputSize(), outputSize());

			for(int j = 0; j < outputSize(); j++){
				for(int i = 0; i < inputSize(); i++){
					float value1 = weights -> getValue(i, j);
					float value2 = b -> weights -> getValue(i, j);

					float rand = random();
					ret -> weights -> setValue(i, j, rand*value1 + (1 - rand)*value2);

				}

				float value1 = biases -> getValue(0, j);
				float value2 = b -> biases -> getValue(0, j);

				float rand = random();
				ret -> weights -> setValue(0, j, rand*value1 + (1 - rand)*value2);

			}

			return ret;

		}else{
			throw std::invalid_argument("Combining layers of different sizes!");
		}

	}

	int Dense::inputSize() const{
		return weights -> width;
	}

	int Dense::outputSize() const{
		return weights -> height;
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

	Layer* Network::getLayer(uint index){
		return layers[index];
	}

	int Network::getNetworkSize(){
		return layers.size();
	}


	Matrix Network::eval(const Matrix* input) {
		Matrix values = dereference(input);

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



	//miscellaneous functions

	//mutate network of dense layers
	void mutate(Network * network, float min, float max){
		for(int i = 0; i < network -> getNetworkSize(); i++){
			try{
				Layer* layer = network -> getLayer(i);
				if(layer -> type == "Dense"){
					((Dense*) layer) -> mutate(min, max);
					std::cout << "Mutated " << i << std::endl;
				}else{
					throw i + 1;
				}
			}catch (int exc){
				std::cout << "Layer " << exc << " is not dense!" << std::endl;
			}
		}
	}

	//comparator function (useful for sorting pairs <network, score>)
	bool compare(const std::pair<Network*, float> lhs, const std::pair<Network*, float> rhs){
		return lhs.second < rhs.second;
	}

	//run genetic learning algorithm with given evaluation function
	//tries to minimize given evaluation function
	Network** genetic(Network** networks, void (*eval)(uint, Network**, float*), gen_settings settings){

		uint population = settings.population;
		uint generations = settings.generations;
		uint mutations = settings.mutations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		float mmin = settings.min;
		float mmax = settings.max;

		float scores[population];


		for(uint i = 0; i < generations; i++){
			std::clock_t start_time;
			start_time = std::clock();

			if(settings.output){
				std::cout << "---- Generation " << i << " ----" << std::endl;
			}


			//run evaluation function
			if(settings.recompute_parents || i == 0){
				for(uint s = 0; s < population; s++){
					scores[s] = 0;
				}

				eval(population, networks, scores);
			}else{
				for(uint s = parent_population; s < population; s++){
					scores[s] = 0;
				}

				eval(population - parent_population, networks + parent_population, scores + parent_population);
			}

			//sort values
			std::vector<std::pair<Network*, float>> pairs(population);
			for(uint j = 0; j < population; j++){
				pairs[j] = std::make_pair(networks[j], scores[j]);
			}

			std::sort(pairs.begin(), pairs.end(), compare);

			for(uint j = 0; j < population; j++){
				Network* evaluate(Network * networks, float* (*func)(uint, Network*), uint population, uint generations);
				networks[j] = pairs[j].first;
				scores[j] = pairs[j].second;
			}


			//remove part of the population and repopulate
			for(uint j = parent_population; j < population; j++){
				//delete network
				delete networks[j];

				//create a new child
				networks[j] = new nnlib::Network();

				//choose random parents
				int parent1 = nnlib::randomInt(0, parent_population - 1);
				int parent2 = nnlib::randomInt(0, parent_population - 1);

				for(int l = 0; l < networks[parent1] -> getNetworkSize(); l++){
					nnlib::Dense * layer1 = ((nnlib::Dense *)(networks[parent1] -> getLayer(l)));
					nnlib::Dense * layer2 = ((nnlib::Dense *)(networks[parent2] -> getLayer(l)));

					nnlib::Dense * layer_child = layer1 -> crossover(layer2);

					networks[j] -> addLayer(layer_child);

					//mutate layer
					for(uint m = 0; m < mutations; m++){
						layer_child -> mutate(mmin, mmax);
					}
				}
			}

			std::clock_t end_time;
			end_time = std::clock();
			if(settings.output){
				printf(" Best score: %.2f\n", scores[0]);
				printf(" Computation time: %.2fs\n", (end_time - start_time)/(double)CLOCKS_PER_SEC);
			}
		}

		return networks;

	}


}
