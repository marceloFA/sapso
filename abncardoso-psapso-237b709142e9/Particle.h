#pragma once

#include <vector>
#include <time.h>

#include "Rand.h"

class Particle
{
private:
	std::vector<double> position;
	std::vector<double> personalBestPosition;
	std::vector<double> velocity;
	std::vector<double> gradient;
	std::vector<int> personalDecision;
	double fitness;
	double personalBestFitness;
	double oldFitness;

public:
	std::vector<double> getPosition();
	void setPosition(std::vector<double> position);
	void updatePosition(std::vector<double> velocity);

	std::vector<double> getVelocity();
	void setVelocity(std::vector<double> velocity);

	double getFitness();
	void setFitness(double fitness);

	std::vector<double> getGradient();
	void setGradient(std::vector<double> gradient);

	/*std::vector<int> Particle::getPersonalDecision();
	void Particle::setPersonalDecision(std::vector<int> newPersonalDecision);*/

	std::vector<double> getPersonalBestPosition();
	void setPersonalBestPosition(std::vector<double> newPersonalBest);

	double getPersonalBestFitness();
	void setPersonalBestFitness(double newPersonalBest);

	double getOldFitness();
	void setOldFitness(double newValue);
};
