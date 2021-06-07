#include "matrix.h"


namespace nnlib {

	// abstract Layer class: not to be implemented
	class Layer {
	public:

		virtual std::string serialize();
		virtual void deserialize(std::string input);

		virtual void eval(Matrix* input, Matrix* output);

		virtual Layer* clone();

	};

	class Dense: public Layer {
	public:

		Dense(uint input, uint output);

		std::string serialize();
		void deserialize(std::string input);

		void eval(Matrix* input, Matrix* output);

		Layer* clone();

	private:
		Matrix* matrix;

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
