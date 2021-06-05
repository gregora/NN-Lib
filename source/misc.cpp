#include "../include/misc.h"
#include <random>

namespace nnlib {

	float random() {

		std::random_device rd{};
		std::mt19937 engine{rd()};
		std::uniform_real_distribution<double> dist{0.0, 1.0};
		double ret = dist(engine);
		return ret;

	}

	float random(float min, float max) {

		return random()*(max-min) + min;

	}

	int numlen(int x) {
		return floor(log10(x)) + 1;
	}
}
