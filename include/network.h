#include "matrix.h"
#include "misc.h"
#include <iostream>
#include <algorithm>
#include <ctime>


namespace nnlib {

	// abstract Layer class: not to be implemented
	class Layer {
	public:
		std::string type;

		virtual std::string serialize() = 0;
		virtual void deserialize(std::string input) = 0;

		virtual Matrix eval(const Matrix* input) = 0;

		virtual Layer* clone() = 0;

		std::string getName() const;
		void setName(std::string new_name);

		//potentially add input/output sizes for addLayer() function in Network class

		virtual ~Layer(){};

	private:
		std::string name;
	};

	class Dense: public Layer {
	public:

		Dense(uint input, uint output, std::string name = "dense_layer");


		std::string serialize();
		void deserialize(std::string input);

		Matrix eval(const Matrix* input);

		Layer* clone();

		void setActivationFunction(std::string name);

		void randomize(float min, float max);
		void mutate(float delta);
		Dense * crossover(const Dense * b) const;
		Dense * crossover_avg(const Dense * b) const;

		~Dense();

		int inputSize() const;
		int outputSize() const;

		Matrix* weights;
		Matrix* biases;

	private:
		float (*activationFunction)(float) = &fast_sigmoid;
		std::string activationFunctionName = "fast_sigmoid";
	};





	class Network {
	public:
		Network(std::string name = "network");

		void addLayer(Layer* l);
		Layer* getLayer(uint index);
		int getNetworkSize();

		Matrix eval(const Matrix* input);

		std::string serialize();
		void deserialize(std::string input);

		void save();
		void save(std::string path);
		void load(std::string path);

		// pretty-print; not for exporting (could be renamed to debug())
		void print();

		Network* clone();

		std::string getName() const;
		void setName(std::string new_name);

		~Network();

	private:
		std::vector<Layer*> layers;
		std::string name;
	};

}
