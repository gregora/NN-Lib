#include "../include/algorithms.h"

namespace nnlib {

	//clock
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	using std::chrono::time_point;

	float passedTime(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end){
		return (float)(duration_cast<milliseconds>(end - start)).count() / 1000;
	}

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

		std::thread* threads = new std::thread[population - parent_population];

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

		delete[] threads;
	}

	//assumes evaluation of all networks in one call
	//run genetic learning algorithm with given evaluation function
	//tries to minimize given evaluation function
	Network** genetic(Network** networks, void (*eval)(uint, Network**, float*), gen_settings settings){

		uint population = settings.population;
		uint generations = settings.generations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		float* scores = new float[population];

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
				printf(" Evaluation:    %10.2f s\n", passedTime(start_time_2, high_resolution_clock::now()));
			}

			//sort networks
			sort(population, networks, scores);

			//repopulate
			start_time_2 = high_resolution_clock::now();
			repopulate(networks, settings);

			if(settings.output){
				printf(" Repopulation:  %10.2f s\n", passedTime(start_time_2, high_resolution_clock::now()));
			}


			if(settings.output){
				printf(" Overall:       %10.2f s\n", passedTime(start_time, high_resolution_clock::now()));
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

		delete[] scores;
		return networks;

	}


	//assumes evaluation of each network individually
	Network** genetic(Network ** networks, void (*eval)(Network*, float *), gen_settings settings){
		uint population = settings.population;
		uint generations = settings.generations;

		uint parent_population = (uint) (((float) population) * settings.rep_coef) ;

		float* scores = new float[population];


		for(uint i = 0; i < generations; i++){
			auto start_time = high_resolution_clock::now();
			auto start_time_2 = start_time;

			if(settings.output){
				printf("---- Generation %d ----\n\n", i + settings.start_generation);
			}

			//run evaluation function
			start_time_2 = high_resolution_clock::now();
			if(settings.recompute_parents || i == 0){

				std::thread* threads = new std::thread[population];

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

				delete[] threads;

			}else{
				std::thread* threads = new std::thread[population - parent_population];

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

				delete[] threads;

			}
			if(settings.output){
				printf(" Evaluation:    %10.2f s\n", passedTime(start_time_2, high_resolution_clock::now()));
			}

			//sort networks
			sort(population, networks, scores);

			//repopulate
			start_time_2 = high_resolution_clock::now();
			repopulate(networks, settings);

			if(settings.output){
				printf(" Repopulation:  %10.2f s\n", passedTime(start_time_2, high_resolution_clock::now()));
				printf(" Overall:       %10.2f s\n", passedTime(start_time, high_resolution_clock::now()));
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

		delete[] scores;
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

	std::vector<deltas> getDeltas(Network * network, const Matrix * target, float speed){
		uint size = network -> getNetworkSize();

		std::vector<deltas> arr;
		arr.resize(size);

		Matrix tar = dereference(target);

		for(uint i = 0; i < size; i++){

			//printf("%d\n", i);
			Dense* layer = (Dense*) network -> getLayer(size - i - 1);

			arr.at(size - i - 1) = layer -> getDeltas(&tar);

			tar = dereference(layer -> input) - dereference((arr.at(size - i - 1)).input)*speed;

		}

		return arr;
	}

	void fit(Network * network, std::vector<Matrix*> input, std::vector<Matrix*> target, fit_settings settings){
		printf("\n\n");

		uint epochs = settings.epochs;
		uint batch_size = settings.batch_size;
		float speed = settings.speed;

		auto start_time = high_resolution_clock::now();

		std::vector<deltas> delt[batch_size];

		for(uint i = 1; i <= epochs; i++){

			//run backpropagation
			float cost = 0;

			uint batch_index = 0;

			for(uint d = 0; d < input.size(); d++){

				Matrix out = network -> eval(input[d]);
				cost += meanSquaredError(&out, target[d]);
				//backpropagate(network, target[d], speed);

				delt[batch_index] = getDeltas(network, target[d], speed);


				batch_index++;

				if(batch_index % batch_size == 0 || d + 1 == input.size()){

					uint network_size = network -> getNetworkSize();
					for(uint i = 0; i < network_size; i++){
						Dense* layer = (Dense*) network -> getLayer(i);

						Matrix delta_weights(layer -> inputSize(), layer -> outputSize());
						Matrix delta_biases(1, layer -> outputSize());

						delta_weights.fillZero();
						delta_biases.fillZero();

						for(uint j = 0; j < batch_index; j++){
							delta_weights = delta_weights + dereference(delt[j][i].weights)*(1.0f / (batch_index));
							delta_biases = delta_biases + dereference(delt[j][i].biases)*(1.0f / (batch_index));

							delete delt[j][i].weights;
							delete delt[j][i].biases;
							delete delt[j][i].input;
						}

						deltas d = {
							weights: &delta_weights,
							biases: &delta_biases
						};

						layer -> applyDeltas(d, speed);
					}
					batch_index = 0;
				}
			}

			//output
			printf("============================\n\n");
			printf(" Epoch:   %10d\n", i);

			cost = cost / input.size();
			printf(" Elapsed: %10.2f s\n", passedTime(start_time, high_resolution_clock::now()));
			printf(" Cost:    %10.5f\n", cost);

			printf(" [");
			for(float p = 0.1; p <= 2; p += 0.1){
				if(p/2 <= (float) i / epochs){
					printf("=");
				}else{
					printf(" ");
				}
			}
			printf("] %3d%%\n", (int) ((float) i*100 / epochs));

			printf("\n============================\n\n");

			if(strcmp(settings.output.c_str(), "minimal") == 0 && i != epochs){
				printf("\x1B[9A");
			}
		}

	}
}
