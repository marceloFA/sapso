#pragma once

#include "Services.h"

class SAPSOServices : public Services
{
private:
	Rand randomizer;

public:
    SAPSOServices(std::string functionName, int aux) : Services(functionName) { randomizer.SetSeed(aux*time(NULL)); };

	void updateVelocity(Particle &particle, std::vector<double> globalBest, int decision, float iw, float sc, double gc, int dir, const float VMAX);
	void updateDir(int &dir, double diversity, std::vector<int> &particleDecision, const double *DT, int swarmSize);
	void truncSpaceSAPSO(Particle &particle, int &decision, int &counter, const float* RANGE);
	void updateImportance(std::vector<Particle> swarm, std::vector<int> &decision, std::vector<int> &counter, const int CMAX, int globalBest);

};
