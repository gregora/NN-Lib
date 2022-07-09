#include "include/algorithms.h"
#include <iostream>
#include <math.h>

using namespace nnlib;

void evaluate(uint size, Network** networks, float* scores){

	Matrix input(1, 3);
	input.setValue(0, 0, 0);
	input.setValue(0, 1, 1);
	input.setValue(0, 2, 0);

	for(uint i = 0; i < size; i++){
		float score = 0;
		for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
			input.setValue(0, 0, x);
			Matrix res = networks[i] -> eval(&input);
			score += abs(res.getValue(0, 0) - sin(x) - 1);
		}
		scores[i] = score;

	}

}

void evaluate_single(Network* network, float* score){

		Matrix input(1, 1);

		*score = 0;

		for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
			input.setValue(0, 0, x);
			Matrix res = network -> eval(&input);
			*score += abs(res.getValue(0, 0) - sin(x) - 1);
		}

}

int main() {

	//sine curve approximation with a neural network

	int POPULATION = 100;
	int GENERATIONS = 200;
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

		layer1 -> randomize(-1, 1);
		layer3_1 -> randomize(-1, 1);
		layer3_2 -> randomize(-1, 1);
		layer3_3 -> randomize(-1, 1);
		layer3_4 -> randomize(-1, 1);
		layer3_5 -> randomize(-1, 1);
		layer5 -> randomize(-1, 1);

		layer1 ->  setActivationFunction(tanh);
		layer3_1 ->  setActivationFunction(tanh);
		layer3_2 ->  setActivationFunction(tanh);
		layer3_3 ->  setActivationFunction(tanh);
		layer3_4 ->  setActivationFunction(tanh);
		layer3_5 ->  setActivationFunction(tanh);
		layer5 ->  setActivationFunction(linear);

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
		mutations: 1, //number of mutations on each child
		rep_coef: 0.1, //percent of population to reproduce
		min: -1, //minimum value for weights / biases
		max: 1, //maximum value for weights / biases
		recompute_parents: false, //recompute parents (for non-deterministic evaluation functions)
		multithreading: true,
		//output settings
		output: true,
		start_generation: 1
	};


	//genetic(networks, evaluate, settings);
	genetic(networks, evaluate_single, settings);

	std::cout << networks[0] -> serialize() << std::endl;

	Matrix input(1, 1);

	for(float x = -2*3.14; x < 2*3.14; x+=4*3.14/20){
		input.setValue(0, 0, x);
		Matrix res = networks[0] -> eval(&input);
		for(int p = 0; p - 30 < (res.getValue(0,0))*10; p++){
			printf(" ");
		}
		printf("%2.1f\n", res.getValue(0,0));
	}

	return 0;
}
