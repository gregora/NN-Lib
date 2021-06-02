#include "matrix.h"


namespace nnlib {

	// abstract Layer class: not to be implemented
	class Layer {
	public:
		virtual std::string toString() = 0;

		virtual std::string serialize() = 0;
		virtual std::string deserialize() = 0;

		virtual float eval(uint datac, float* datav);

	};

	class Dense: public Layer {
	public:
		std::string toString();

		std::string serialize();
		std::string deserialize();

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
		std::string deserialize();

		// pretty-print; not for exporting
		std::string toString();

		Network clone();

	private:
		Layer* layers;

	};
}
