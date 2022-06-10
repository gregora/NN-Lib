#include "include/network.h"
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

	}

}

int main() {

	int POPULATION = 100000;
	int GENERATIONS = 10000;
	Network* networks[POPULATION];

	for(int i = 0; i < POPULATION; i++){
		networks[i] = new Network();
		Dense* layer = new Dense(3, 3);
		layer -> randomize(0, 1);
		networks[i] -> addLayer(layer);
	}

	gen_settings settings = {
		population: POPULATION,
		generations: GENERATIONS,
		mutations: 1,
		rep_coef: 0.9,
		min: 0,
		max: 1,
		recompute_parents: false,
	};

	genetic(networks, evaluate, settings);

	std::cout << networks[0] -> serialize() << std::endl;

	Matrix input(1, 3);
	input.setValue(0, 0, 0);
	input.setValue(0, 1, 1);
	input.setValue(0, 2, 0);
	Matrix res = networks[0] -> eval(&input);
	std::cout << res.serialize() << std::endl;

	return 0;
}
