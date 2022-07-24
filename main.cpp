#include "include/algorithms.h"
#include <iostream>
#include <math.h>

using namespace nnlib;

void evaluate(uint size, Network** networks, float* scores){

	Matrix input(1, 1);

	for(uint i = 0; i < size; i++){
		float score = 0;
		for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
			input.set(0, 0, x);
			Matrix res = networks[i] -> eval(&input);
			score += abs(res.get(0, 0) - sin(x));
		}
		scores[i] = score / 20;

	}

}

void evaluate_single(Network* network, float* score){

		Matrix input(1, 1);

		*score = 0;

		for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
			input.set(0, 0, x);
			Matrix res = network -> eval(&input);
			*score += abs(res.get(0, 0) - sin(x));
		}

}

int train_genetic() {
	//approximating sine curve with a neural network using genetic algorithm

	int POPULATION = 100;
	int GENERATIONS = 50;
	Network* networks[POPULATION];

	for(int i = 0; i < POPULATION; i++){
		networks[i] = new Network();
		Dense* layer1 = new Dense(1, 10);
		Dense* layer3_1 = new Dense(10, 10);
		Dense* layer3_2 = new Dense(10, 10);
		Dense* layer3_3 = new Dense(10, 10);
		Dense* layer3_4 = new Dense(10, 10);
		Dense* layer3_5 = new Dense(10, 10);
		Dense* layer5 = new Dense(10, 1);

		layer1 ->  setActivationFunction("tanh");
		layer3_1 ->  setActivationFunction("tanh");
		layer3_2 ->  setActivationFunction("tanh");
		layer3_3 ->  setActivationFunction("tanh");
		layer3_4 ->  setActivationFunction("tanh");
		layer3_5 ->  setActivationFunction("tanh");
		layer5 ->  setActivationFunction("linear");

		networks[i] -> addLayer(layer1);
		networks[i] -> addLayer(layer3_1);
		networks[i] -> addLayer(layer3_2);
		networks[i] -> addLayer(layer3_3);
		networks[i] -> addLayer(layer3_4);
		networks[i] -> addLayer(layer3_5);
		networks[i] -> addLayer(layer5);

	}

	gen_settings settings = {
		//general settings
		population: POPULATION,
		generations: GENERATIONS, //number of generations to run
		mutation_rate: 0.1, //number of mutations on each child
		rep_coef: 0.1, //percent of population to reproduce
		delta: 0.2,
		recompute_parents: false, //recompute parents (for non-deterministic evaluation functions)
		multithreading: true,

		//file saving settings
		save_period: 100, //how often networks are saved
		path: "saves/", //empty folder for saving
		start_generation: 0,

		//output settings
		output: true
	};

	std::cout << "Train the network ...\n\n\n\n";

	//genetic(networks, evaluate, settings);
	genetic(networks, evaluate_single, settings);

	std::cout << "Network output (sin curve approximation) ...\n\n\n\n";

	Matrix input(1, 1);

	for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
		input.set(0, 0, x);
		Matrix res = networks[0] -> eval(&input);
		for(int p = 0; p - 30 < (res.get(0,0))*10; p++){
			printf(" ");
		}
		printf("%2.1f\n", res.get(0,0));
	}

	std::cout << "\n\nSave the network as sin.AI ...\n\n\n\n";
	networks[0] -> save("sin.AI");


	std::cout << "\nLoad the network ...\n\n\n\n";

	Network nnsin;
	nnsin.load("sin.AI");
	for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
		input.set(0, 0, x);
		Matrix res = nnsin.eval(&input);
		for(int p = 0; p - 30 < (res.get(0,0))*10; p++){
			printf(" ");
		}
		printf("%2.1f\n", res.get(0,0));
	}

	return 0;
}







int train_backpropagation(){
	//approximating sine curve with a neural network using backpropagation

	//create dataset
	std::vector<Matrix*> input;
	std::vector<Matrix*> target;

	for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
		Matrix* i = new Matrix(1, 1);
		i -> set(0, 0, x);

		Matrix* o = new Matrix(1, 1);
		o -> set(0, 0, sin(x));

		input.push_back(i);
		target.push_back(o);
	}

	//build network
	Network n;
	Dense* layer1 = new Dense(1, 100);
	Dense* layer4 = new Dense(100, 1);

	layer1 -> setActivationFunction("relu");
	layer4 -> setActivationFunction("linear");

	n.addLayer(layer1);
	n.addLayer(layer4);

	//fit the model

	fit_settings settings = {
		epochs: 20000,
		speed: 0.001,

		output: "minimal"
	};

	fit(&n, input, target, settings);

	//output
	Matrix in(1,1);
	for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
		in.set(0, 0, x);
		Matrix res = n.eval(&in);
		for(int p = 0; p - 30 < (res.get(0,0))*10; p++){
			printf(" ");
		}
		printf("%2.1f\n", res.get(0,0));
	}

}

int main(){
	train_backpropagation();
}
