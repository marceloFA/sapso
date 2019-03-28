#include "SAPSOServices.h"

void SAPSOServices::updateVelocity(Particle &particle, std::vector<double> globalBest, int decision, float iw, float sc, double gc, int dir, const float VMAX)
{
	std::vector<double> velocity = particle.getVelocity();
	std::vector<double> personalBestPosition = particle.getPersonalBestPosition();
	std::vector<double> currentPosition = particle.getPosition();
	std::vector<double> gradient = particle.getGradient();
	
	for (int i = 0; i < velocity.size(); i++)
	{
		velocity[i] = iw * velocity[i] + dir * (decision * sc * randomizer.Uniform() * (globalBest[i] - currentPosition[i]) +
			(decision - 1) * gc * randomizer.Uniform() * gradient[i]);
	}

	velocity = trunc(velocity, VMAX);

	particle.setVelocity(velocity);
}

void SAPSOServices::updateDir(int &dir, double diversity, std::vector<int> &particleDecision, const double *DT, int swarmSize)
{
	if (dir > 0 && diversity < DT[0])
	{
		dir = -1;
		particleDecision = std::vector<int>(swarmSize, 1);
	}
	else if (dir < 0 && diversity > DT[1])
	{
		dir = 1;
		particleDecision = std::vector<int>(swarmSize, 0);
	}
}

void SAPSOServices::truncSpaceSAPSO(Particle &particle, int &decision, int &counter, const float* RANGE)
{
	std::vector<double> position = particle.getPosition();
	bool decisionFlag = false;

	for (int i = 0; i < position.size(); i++)
	{
		if (position[i] < RANGE[0])
		{
			position[i] = RANGE[0];
			decisionFlag = true;
		}
		else if (position[i] > RANGE[1])
		{
			position[i] = RANGE[1];
			decisionFlag = true;
		}
	}

	if (decisionFlag)
	{
		decision = 1;
		counter = 0;
	}

	particle.setPosition(position);
}

void SAPSOServices::updateImportance(std::vector<Particle> swarm, std::vector<int> &decision, std::vector<int> &counter, const int CMAX, int globalBest)
{
	int swarmSize = swarm.size();
	double fitness;
	double oldFitness;
	double norm;
	double sum;
	std::vector<double> position;
	std::vector<double> gradient;
	std::vector<double> globalBestPosition = swarm[globalBest].getPersonalBestPosition();
	int dimensionSize = globalBestPosition.size();

	for (int i = 0; i < swarmSize; i++)
	{
		position = swarm[i].getPosition();
		fitness = swarm[i].getFitness();
		oldFitness = swarm[i].getOldFitness();
		gradient = swarm[i].getGradient();
		sum = 0;
		norm = 0;

		for (int j = 0; j < dimensionSize; j++)
		{
			norm += std::pow(gradient[j], 2);
			sum += std::pow((position[j] - globalBestPosition[j]), 2);
		}
		norm = std::sqrt(norm);
		sum = std::sqrt(sum);

		if (decision[i] == 0)
		{
			if (std::abs(fitness - oldFitness) < 1e-2 || norm < 1e-2)
			{
				counter[i]++;
				if (counter[i] == CMAX)
				{
					decision[i] = 1;
					counter[i] = 0;
				}
			}
			else
			{
				counter[i] = 0;
			}
		}
		else
		{
			if (sum < 1e-5)
			{
				decision[i] = 0;
				counter[i] = 0;
			}
		}
	}
}