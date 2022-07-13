#include "../include/algorithms.h"

namespace nnlib {

	//clock
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	//mutate network of dense layers
	void mutate(Network * network, float delta){
		for(int i = 0; i < network -> getNetworkSize(); i++){
			try{
				Layer* layer = network -> getLayer(i);
				if(layer -> type == "Dense"){
					((Dense*) layer) -> mutate(delta);
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

	//create a child from parents
	void create_child(Network * parent1, Network * parent2, Network ** child_p, gen_settings settings){

		float delta = settings.delta;
		float mutations = settings.mutation_rate;

		delete *child_p;

		//create a new child
		Network* child = new nnlib::Network();

		for(int l = 0; l < parent1 -> getNetworkSize(); l++){
			nnlib::Dense * layer1 = ((nnlib::Dense *)(parent1 -> getLayer(l)));
			nnlib::Dense * layer2 = ((nnlib::Dense *)(parent2 -> getLayer(l)));

			nnlib::Dense * layer_child = layer1 -> crossover(layer2);

			child -> addLayer(layer_child);

			//mutate layer
			float r = random();
			//printf("%f\n", r);
			if(r <= mutations){
				layer_child -> mutate(delta);
			}

		}

		*child_p = child;

	}

	//repopulate with children
	//assumes array: [parent1, parent2, ...., parent n, child 1, child 2, ...]
	void repopulate(Network ** networks, gen_settings settings){
		uint population = settings.population;
		uint mutations = settings.mutation_rate;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		std::thread threads[population - parent_population];

		//remove part of the population and repopulate
		for(uint i = parent_population; i < population; i++){
			//choose random parents
			int parent1 = nnlib::randomInt(0, parent_population - 1);
			int parent2 = nnlib::randomInt(0, parent_population - 1);

			if(!settings.multithreading){
				//single thread
				create_child(networks[parent1], networks[parent2], &(networks[i]), settings);
			}else{
				threads[i - parent_population] = std::thread(create_child, networks[parent1], networks[parent2], &(networks[i]), settings);
			}
		}

		//join threads
		if(settings.multithreading){
			for(uint i = parent_population; i < population; i++){
				threads[i - parent_population].join();
			}
		}

	}

	//assumes evaluation of all networks in one call
	//run genetic learning algorithm with given evaluation function
	//tries to minimize given evaluation function
	Network** genetic(Network** networks, void (*eval)(uint, Network**, float*), gen_settings settings){

		uint population = settings.population;
		uint generations = settings.generations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		float scores[population];


		for(uint i = 0; i < generations; i++){
			auto start_time = high_resolution_clock::now();
			auto start_time_2 = start_time;

			if(settings.output){
				printf("---- Generation %d ----\n\n", i + settings.start_generation);
			}


			//run evaluation function
			start_time_2 = high_resolution_clock::now();
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
				printf(" Evaluation:    %10.2f s\n", (float)(duration_cast<milliseconds>(high_resolution_clock::now() - start_time_2)).count() / 1000);
			}

			//sort networks
			sort(population, networks, scores);

			//repopulate
			start_time_2 = high_resolution_clock::now();
			repopulate(networks, settings);

			if(settings.output){
				printf(" Repopulation:  %10.2f s\n", (float)(duration_cast<milliseconds>(high_resolution_clock::now() - start_time_2)).count() / 1000);
			}


			auto end_time = high_resolution_clock::now();
			if(settings.output){
				printf(" Overall:       %10.2f s\n", (float)(duration_cast<milliseconds>(end_time - start_time)).count() / 1000);
				printf("\n Best score:    %10.2f\n", scores[0]);
				printf("\n\n");
			}
		}

		return networks;

	}


	//assumes evaluation of each network individually
	Network** genetic(Network ** networks, void (*eval)(Network*, float *), gen_settings settings){
		uint population = settings.population;
		uint generations = settings.generations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		float scores[population];


		for(uint i = 0; i < generations; i++){
			auto start_time = high_resolution_clock::now();
			auto start_time_2 = start_time;

			if(settings.output){
				printf("---- Generation %d ----\n\n", i + settings.start_generation);
			}

			//run evaluation function
			start_time_2 = high_resolution_clock::now();
			if(settings.recompute_parents || i == 0){

				std::thread threads[population];

				for(uint s = 0; s < population; s++){
					//reset scores
					scores[s] = 0;

					if(!settings.multithreading){
						eval(networks[s], scores + s);
					}else{
						threads[s] = std::thread(eval, networks[s], scores + s);
					}
				}

				if(settings.multithreading){
					for(uint s = 0; s < population; s++){
						threads[s].join();
					}
				}

			}else{
				std::thread threads[population - parent_population];

				//reset scores
				for(uint s = parent_population; s < population; s++){
					scores[s] = 0;

					if(!settings.multithreading){
						eval(networks[s], scores + s);
					}else{
						threads[s - parent_population] = std::thread(eval, networks[s], scores + s);
					}
				}

				if(settings.multithreading){
					for(uint s = parent_population; s < population; s++){
						threads[s - parent_population].join();
					}
				}

			}
			if(settings.output){
				printf(" Evaluation:    %10.2f s\n", (float)(duration_cast<milliseconds>(high_resolution_clock::now() - start_time_2)).count() / 1000);
			}

			//sort networks
			sort(population, networks, scores);

			//repopulate
			start_time_2 = high_resolution_clock::now();
			repopulate(networks, settings);

			if(settings.output){
				printf(" Repopulation:  %10.2f s\n", (float)(duration_cast<milliseconds>(high_resolution_clock::now() - start_time_2)).count() / 1000);
			}


			auto end_time = high_resolution_clock::now();
			if(settings.output){
				printf(" Overall:       %10.2f s\n", (float)(duration_cast<milliseconds>(end_time - start_time)).count() / 1000);
				printf("\n Best score:    %10.2f\n", scores[0]);
				printf("\n\n");
			}
		}

		return networks;
	}

}
