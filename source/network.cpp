#include "../include/network.h"

namespace nnlib {

	Dense::Dense(uint input, uint output){
		weights = new Matrix(input, output);
		weights -> fillRandom();

		biases = new Matrix(1, output);
		biases -> fillRandom();
	}

	Matrix Dense::eval(Matrix input){

		return (*weights) * input + (*biases);

	}


	std::string Dense::serialize(){
		return NULL;

	}
	void Dense::deserialize(std::string input){
		
	}

	Layer* Dense::clone(){
		return NULL;
	}

}
