#include "../include/network.h"

namespace nnlib {

	Dense::Dense(uint input, uint output, std::string activation, std::string name) {

		type = "Dense";

		weights = new Matrix(input, output);
		weights -> fillRandom(-1, 1);

		biases = new Matrix(1, output);
		biases -> fillRandom(-1, 1);

		this -> input = new Matrix(1, input);
		this -> logits = new Matrix(1, output);
		this -> output = new Matrix(1, output);

		setName(name);
		weights -> setName(name + "_weights");
		biases -> setName(name + "_biases");

		setActivationFunction(activation);
	}

	Matrix Dense::eval(const Matrix* input) {

		Matrix& inp = *(this -> input);
		inp = dereference(input);

		Matrix& weights = *(this -> weights);
		Matrix& biases = *(this -> biases);
		Matrix& logits = *(this -> logits);
		Matrix& output = *(this -> output);

		Matrix ret = weights * inp + biases;

		logits = ret;
		ret = activationFunction(ret);
		output = ret;

		return ret;

	}

	deltas Dense::getDeltas(const Matrix * target) const{

		if(target -> height != output -> height || target -> width != 1){
			throw std::invalid_argument("Target matrix is of incorrect shape (h=" + std::to_string(target -> height) + " w=" + std::to_string(target -> width) + ")");
		}

		Matrix* weight_deltas = new Matrix(weights -> width, weights -> height);
		Matrix* bias_deltas = new Matrix(1, biases -> height);
		Matrix* input_deltas = new Matrix(1, input -> height);

		deltas deltas;
		deltas.weights = weight_deltas;
		deltas.biases = bias_deltas;
		deltas.input = input_deltas;

		Matrix error_jacobian = errorFunctionDerivative(*output, *target);
		Matrix activation_jacobian = activationFunctionDerivative(*output);

		//precalculate k values
		Matrix k(output -> height, 1); //k is equal to J E * J Act

		for(uint j = 0; j < k.width; j++){
			float value = 0;

			for(uint j_ = 0; j_ < k.width; j_++){
				value += - error_jacobian.get(0, j_) * activation_jacobian.get(j, j_);
			}

			k.set(j, 0, value);
		}

		//calculate delta weights
		for(uint i = 0; i < weights -> width; i++){

			float x_i = input -> get(0, i);

			for(uint j = 0; j < weights -> height; j++){
				float delta = k.get(j, 0) * x_i;
				weight_deltas -> set(i, j, delta);
			}
		}

		//calculate delta biases
		*bias_deltas = k.transpose();

		//calculate delta target
		*input_deltas = (k * (*weights)).transpose();

		return deltas;
	}

	void Dense::applyDeltas(deltas deltas, float speed){


		Matrix& delta_weights = *(deltas.weights);
		Matrix& weights = *(this -> weights);

		Matrix& delta_biases = *(deltas.biases);
		Matrix& biases = *(this -> biases);

		//apply weights
		weights = weights - delta_weights*speed;

		//apply biases
		biases = biases - delta_biases*speed;

	}

	Matrix Dense::backpropagate(const Matrix* target, float speed){

		//get deltas and apply them
		deltas deltas = getDeltas(target);
		applyDeltas(deltas, speed);

		Matrix ret = dereference(deltas.input);

		for(uint j = 0; j < ret.height; j++){
			ret.set(0, j, input -> get(0, j) - speed * ret.get(0, j) );
		}

		delete deltas.weights;
		delete deltas.biases;
		delete deltas.input;

		return ret;
	}


	std::string Dense::serialize() {
		return getName() + "\n===\n" + activationFunctionName + "\n===\n" + weights -> serialize(7) + "===\n" + biases -> serialize(7);
	}

	void Dense::deserialize(std::string input) {
		std::vector<std::string> split = splitString(input, "\n===\n");
		if (split.size() != 4)
			throw std::invalid_argument("Attempting to deserialize an invalid Dense layer format");
		setName(split[0]);
		setActivationFunction(split[1]);
		weights -> deserialize(split[2] + "\n");
		biases -> deserialize(split[3]);


		this -> input = new Matrix(1, weights -> width);
		this -> logits = new Matrix(1, weights -> height);
	}

	Layer* Dense::clone() {
		Layer * copy = new Dense(weights -> width, weights -> height, getName());
		copy -> deserialize(serialize());
		return copy;
	}

	void Dense::setActivationFunction(std::string name) {
		if(name == "fast_sigmoid"){
			activationFunction = fast_sigmoid;
			activationFunctionDerivative = dfast_sigmoid;
		}else if(name == "sigmoid"){
			activationFunction = sigmoid;
			activationFunctionDerivative = dsigmoid;
		}else if(name == "relu"){
			activationFunction = relu;
			activationFunctionDerivative = drelu;
		}else if(name == "linear"){
			activationFunction = linear;
			activationFunctionDerivative = dlinear;
		}else if(name == "atan"){
			activationFunction = nnlib::atan;
			activationFunctionDerivative = datan;
		}else if(name == "tanh"){
			activationFunction = nnlib::tanh;
			activationFunctionDerivative = dtanh;
		}else if(name == "softmax"){
			activationFunction = softmax;
			activationFunctionDerivative = dsoftmax;
		}else{
			throw name + " is not a valid function name";
			return;
		}

		activationFunctionName = name;
	}

	void Dense::setErrorFunction(std::string name) {
		if(name == "MSE"){
			errorFunction = MSE;
			errorFunctionDerivative = dMSE;
		}else if(name == "categoricalCrossentropy"){
			errorFunction = categoricalCrossentropy;
			errorFunctionDerivative = dcategoricalCrossentropy;
		}else{
			throw name + " is not a valid function name";
			return;
		}

		errorFunctionName = name;
	}

	std::string Dense::getErrorFunction(){
		return errorFunctionName;
	}


	void Dense::randomize(float min, float max){
		weights -> fillRandom(min, max);
		biases -> fillRandom(min, max);
	}

	void Dense::mutate(float delta){
		float rand = random();
		int weights_num = weights->width * weights->height;
		int biases_num = biases->width * biases->height;

		if(rand < (float)weights_num / (weights_num + biases_num)){
			//change random weight
			int x = randomInt(0, weights->width - 1);
			int y = randomInt(0, weights->height - 1);
			float value = weights -> get(x, y) + random(-delta, delta);
			weights -> set(x, y, value);
		}else{
			//change random bias
			int x = randomInt(0, biases->width - 1);
			int y = randomInt(0, biases->height - 1);
			float value = biases -> get(x, y) + random(-delta, delta);
			biases -> set(x, y, value);
		}
	}

	Dense* Dense::crossover(const Dense* b) const{
		if(b -> inputSize() == inputSize() && b -> outputSize() == outputSize()){
			Dense* ret = new Dense(inputSize(), outputSize());

			ret -> setActivationFunction(activationFunctionName);

			int g = 0;

			//random crossover start and end
			int crossover_s = randomInt(0, inputSize()*outputSize() + outputSize());
			int crossover_e = randomInt(crossover_s, inputSize()*outputSize() + outputSize());

			for(int j = 0; j < outputSize(); j++){
				for(int i = 0; i < inputSize(); i++){

					if(g >= crossover_s && g <= crossover_e){
						ret -> weights -> set(i, j, weights -> get(i, j));
					}else{
						ret -> weights -> set(i, j, b -> weights -> get(i, j));
					}

					g++;
				}

				if(g >= crossover_s && g <= crossover_e){
					ret -> biases -> set(0, j, biases -> get(0, j));
				}else{
					ret -> biases -> set(0, j, b -> biases -> get(0, j));
				}

				g++;
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
					float value1 = weights -> get(i, j);
					float value2 = b -> weights -> get(i, j);

					float rand = random();
					ret -> weights -> set(i, j, rand*value1 + (1 - rand)*value2);

				}

				float value1 = biases -> get(0, j);
				float value2 = b -> biases -> get(0, j);

				float rand = random();
				ret -> weights -> set(0, j, rand*value1 + (1 - rand)*value2);

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

	Layer* Network::getLayer(uint index) const{
		return layers[index];
	}

	int Network::getNetworkSize() const{
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

	Matrix Network::eval(const Matrix& input){
		return eval(&input);
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
