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
		Matrix res = networks[i] -> eval(&input);
		score += abs(res.getValue(0, 0) - 0.0f);
		score += abs(res.getValue(0, 1) - 1.0f);
		score += abs(res.getValue(0, 2) - 0.0f);
		scores[i] = score;

		for(int j = 0; j < 1000000; j++){
			//simulate long evaluation function
		}

	}

}

void evaluate_single(Network* network, float* score){

		Matrix input(1, 3);
		input.setValue(0, 0, 0);
		input.setValue(0, 1, 1);
		input.setValue(0, 2, 0);

		*score = 0;
		Matrix res = network -> eval(&input);
		*score += abs(res.getValue(0, 0) - 0.0f);
		*score += abs(res.getValue(0, 1) - 1.0f);
		*score += abs(res.getValue(0, 2) - 0.0f);

		for(int j = 0; j < 1000000; j++){
			//simulate long evaluation function
		}
}

int main() {

	int POPULATION = 300;
	int GENERATIONS = 20;
	Network* networks[POPULATION];

	for(int i = 0; i < POPULATION; i++){
		networks[i] = new Network();
		Dense* layer = new Dense(3, 3);
		layer -> randomize(0, 1);
		networks[i] -> addLayer(layer);
	}

	gen_settings settings = {
		//general settings
		population: POPULATION,
		generations: GENERATIONS, //number of generations to run
		mutations: 100, //number of mutations on each child
		rep_coef: 0.5, //percent of population to reproduce
		min: 0, //minimum value for weights / biases
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

	Matrix input(1, 3);
	input.setValue(0, 0, 0);
	input.setValue(0, 1, 1);
	input.setValue(0, 2, 0);
	Matrix res = networks[0] -> eval(&input);
	std::cout << res.serialize() << std::endl;

	return 0;
}
