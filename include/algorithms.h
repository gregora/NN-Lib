#include "matrix.h"
#include "misc.h"
#include "network.h"
#include <sys/stat.h>

namespace nnlib {
	//mutate function for networks with deep layers
	void mutate(Network * network, float delta);

	//run evaluation for n-generations, returns sorted array
	struct gen_settings {
		//general settings
		uint population;
		uint generations = 100; //number of generations to run
		float mutation_rate = 0.1; //number of mutations on each child

		float rep_coef = 0.5; //percent of population to reproduce

		float delta = 0.2; //maximum weight/bias change for mutations

		bool recompute_parents = true; //recompute parents (for non-deterministic evaluation functions)

		bool multithreading = false;

		//file saving settings
		uint save_period = 10; //how often networks are saved (0 == never)
		std::string path = "./"; //empty folder for saving
		uint start_generation = 1; //affects save files

		//output settings
		bool output = true;
	};

	void sort(uint size, Network ** networks, float * scores);

	//assumes evaluation of all networks in one call
	Network** genetic(Network** networks, void (*eval)(uint, Network**, float*), gen_settings settings);

	//assumes evaluation of each network individually
	Network** genetic(Network ** networks, void (*eval)(Network*, float*), gen_settings settings);

	void save_population(Network ** networks, uint population, std::string folder);
	void load_population(Network ** networks, uint population, std::string folder);

}
