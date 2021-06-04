#include "matrix.h"


namespace nnlib {

	// abstract Layer class: not to be implemented
	class Layer {
	public:
		virtual std::string toString() = 0;

		virtual std::string serialize() = 0;
		virtual void deserialize() = 0;

		virtual float eval(Matrix* input, Matrix* output);

		virtual Layer* clone();

	};

	class Dense: public Layer {
	public:
		std::string toString();

		std::string serialize();
		void deserialize();

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
		void deserialize();

		// pretty-print; not for exporting
		std::string toString();

		Network* clone();

	private:
		Layer* layers;

	};
}
