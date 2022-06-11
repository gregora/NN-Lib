#include "../include/algorithms.h"

namespace nnlib {
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

	void sort(uint size, Network ** networks, float * scores){
		//sort values
		std::vector<std::pair<Network*, float>> pairs(size);
		for(uint i = 0; i < size; i++){
			pairs[i] = std::make_pair(networks[i], scores[i]);
		}

		std::sort(pairs.begin(), pairs.end(), compare);

		for(uint i = 0; i < size; i++){
			networks[i] = pairs[i].first;
			scores[i] = pairs[i].second;
		}
	}

	//assumes array: [parent1, parent2, ...., parent n, child 1, child 2, ...]
	void repopulate(Network ** networks, gen_settings settings){
		uint population = settings.population;
		uint mutations = settings.mutations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		float mmin = settings.min;
		float mmax = settings.max;

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
	}

	//run genetic learning algorithm with given evaluation function
	//tries to minimize given evaluation function
	Network** genetic(Network** networks, void (*eval)(uint, Network**, float*), gen_settings settings){

		uint population = settings.population;
		uint generations = settings.generations;
		//uint mutations = settings.mutations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		//float mmin = settings.min;
		//float mmax = settings.max;

		float scores[population];


		for(uint i = 0; i < generations; i++){
			std::clock_t start_time = std::clock();
			std::clock_t start_time_2;

			if(settings.output){
				printf("---- Generation %d ----\n\n", i);
			}

			//run evaluation function
			start_time_2 = std::clock();
			if(settings.recompute_parents || i == 0){
				//reset scores
				for(uint s = 0; s < population; s++){
					scores[s] = 0;
				}

				eval(population, networks, scores);
			}else{
				//reset scores
				for(uint s = parent_population; s < population; s++){
					scores[s] = 0;
				}

				eval(population - parent_population, networks + parent_population, scores + parent_population);
			}
			if(settings.output){
				printf(" Evaluation:    %.2fs\n", (std::clock() - start_time_2)/(double)CLOCKS_PER_SEC);
			}

			//sort networks
			sort(population, networks, scores);

			//repopulate
			start_time_2 = std::clock();
			repopulate(networks, settings);
			if(settings.output){
				printf(" Repopulation:  %.2fs\n", (std::clock() - start_time_2)/(double)CLOCKS_PER_SEC);
			}


			std::clock_t end_time = std::clock();
			if(settings.output){
				printf(" Overall:       %.2fs\n", (end_time - start_time)/(double)CLOCKS_PER_SEC);
				printf("\n Best score:    %.2f\n", scores[0]);
				printf("\n\n");
			}
		}

		return networks;

	}

}
