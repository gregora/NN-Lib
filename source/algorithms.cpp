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


			//save networks every settings.save_period
			if(i == generations - 1 || ((i != 0) && (settings.save_period != 0) && (i % settings.save_period == 0))){
				std::string path(settings.path);
				if(path.back() != '/'){
					path += "/";
				}
				path+="Generation" + std::to_string(i + settings.start_generation);

				save_population(networks, population, path);

				if(settings.output){
					printf("NOTE: Saved networks to %s\n\n\n", path.c_str());
				}
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


			//save networks every settings.save_period
			if(i == generations - 1 || ((i != 0) && (settings.save_period != 0) && (i % settings.save_period == 0))){
				std::string path(settings.path);
				if(path.back() != '/'){
					path += "/";
				}
				path+="Generation" + std::to_string(i + settings.start_generation);

				save_population(networks, population, path);

				if(settings.output){
					printf("NOTE: Saved networks to %s\n\n\n", path.c_str());
				}
			}
		}


		return networks;
	}

	void save_population(Network ** networks, uint population, std::string folder){

		mkdir(folder.c_str(), 0777);

		if(folder.back() != '/'){
			folder += "/";
		}

		for(uint i = 0; i < population; i++){
			networks[i] -> save(folder + std::to_string(i) + ".AI");
		}

	}

	void load_population(Network ** networks, uint population, std::string folder){
		for(uint i = 0; i < population; i++){
			networks[i] = new Network;
			networks[i] -> load(folder + std::to_string(i) + ".AI");
		}
	}


	Matrix backpropagate(Network * network, const Matrix* target, float speed){
		Matrix t = dereference(target);
		try{
			uint size = network -> getNetworkSize();
			for(uint i = 0; i < size; i++){

				Layer* layer = network -> getLayer(size - i - 1);
				if(layer -> type == "Dense"){
					t = ((Dense*)layer) -> backpropagate(&t, speed);
				}else{
					throw i + 1;
				}
			}
		}catch (int exc){
			std::cout << "Layer " << exc << " is not dense!" << std::endl;
		}

		return t;
	}



	void fit(Network * network, std::vector<Matrix*> input, std::vector<Matrix*> target, uint epochs, float speed){

		for(uint i = 0; i < epochs; i++){
			printf("------- Epoch %d -------\n\n", i);

			auto start_time = high_resolution_clock::now();

			float cost = 0;
			for(uint d = 0; d < input.size(); d++){
				Matrix out = network -> eval(input[d]);
				backpropagate(network, target[d], speed);

				for(uint o = 0; o < out.height; o++){
					cost += pow(out.getValue(0, o) - target[d] -> getValue(0, o), 2);
				}
			}

			cost = cost / input.size();
			printf(" Computation:    %10.2f s\n", (float)(duration_cast<milliseconds>(high_resolution_clock::now() - start_time)).count() / 1000);
			printf("\n");
			printf(" Cost: %f\n", cost);
			printf("\n\n");
		}

	}
}
