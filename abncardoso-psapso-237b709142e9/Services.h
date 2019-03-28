#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <cstdlib>
#include <numeric>

#include "EvaluationFunctions.h"
#include "Particle.h"
#include "Rand.h"

class Services
{
private:
	std::map<std::string, std::function<double(std::vector<double>)>> functions;
	Rand randomizer;
	std::string functionName;
public:
	Services(std::string function);

	double evaluateParticle(std::vector<double> particle);
	std::vector<double> generateRandomSet(const int DIM, const float* RANGE);
	int getGlobalBest(std::vector<Particle> swarm);
    int getLessFit(std::vector<Particle> swarm);
	std::vector<double> trunc(std::vector<double> dataSet, const float VMAX);
	void updatePosition(Particle &particle);
	void truncSpace(Particle &particle, const float* RANGE);
	void updatePersonalBest(Particle &particle);
	void getGradient(Particle &particle, const float VMAX);
	double getDiversity(std::vector<Particle> swarm, double diagonalLength, const int swarmSize, const int dimensionSize);
};
