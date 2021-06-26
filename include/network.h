#include "matrix.h"


namespace nnlib {

	// abstract Layer class: not to be implemented
	class Layer {
	public:

		virtual std::string serialize() = 0;
		virtual void deserialize(std::string input) = 0;

		virtual Matrix eval(Matrix input) = 0;

		virtual Layer* clone() = 0;

		std::string getName() const;
		void setName(std::string new_name);

	private:
		std::string name;
	};

	class Dense: public Layer {
	public:

		Dense(uint input, uint output, std::string name = "dense_layer");

		std::string serialize();
		void deserialize(std::string input);

		Matrix eval(Matrix input);

		Layer* clone();

	private:
		Matrix* weights;
		Matrix* biases;

		float (*activation)(float) = nullptr;

	};

	class Network {
	public:
		Network();
		~Network();

		void addLayer(Layer* l);

		float eval(uint datac, float* datav);

		void save(std::string path);
		void load(std::string path);

		std::string serialize();
		void deserialize(std::string input);

		// pretty-print; not for exporting
		std::string toString();

		Network* clone();

	private:
		Layer* layers;

	};
}
