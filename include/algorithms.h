#include "matrix.h"
#include "misc.h"
#include "network.h"

namespace nnlib {
	//mutate function for networks with deep layers
	void mutate(Network * network, float min, float max);

	//run evaluation for n-generations, returns sorted array
	struct gen_settings {
		//general settings
		uint population;
		uint generations = 100; //number of generations to run
		uint mutations = 1; //number of mutations on each child

		float rep_coef = 0.5; //percent of population to reproduce

		float min = 0; //minimum value for weights / biases
		float max = 1; //maximum value for weights / biases

		bool recompute_parents = false; //recompute parents (for non-deterministic evaluation functions)

		bool multithreading = false;

		//output settings
		bool output = true;
		uint start_generation = 1;
	};

	void sort(uint size, Network ** networks, float * scores);

	Network** genetic(Network** networks, void (*eval)(uint, Network**, float*), gen_settings settings);

}
