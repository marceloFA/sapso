#include "Services.h"

Services::Services(std::string function)
{
	functionName = function;

	randomizer.SetSeed(time(NULL));

	functions["sphere"] = &EvaluationFunctions::sphere;
	functions["rosenbrock"] = &EvaluationFunctions::rosenbrock;
	functions["rastrigin"] = &EvaluationFunctions::rastrigin;
	functions["griewank"] = &EvaluationFunctions::griewank;
	functions["ackley"] = &EvaluationFunctions::ackley;
	functions["ellipsoid"] = &EvaluationFunctions::ellipsoid;
	functions["schaffer2"] = &EvaluationFunctions::schaffer2;
	functions["alpine"] = &EvaluationFunctions::alpine;
	functions["levi"] = &EvaluationFunctions::levi;
	functions["levi13"] = &EvaluationFunctions::levi13;
};

double Services::evaluateParticle(std::vector<double> particle)
{
	return functions[functionName](particle);
}

std::vector<double> Services::generateRandomSet(const int DIM, const float* RANGE)
{

	std::vector<double> randomSet;
	for (int i = 0; i < DIM; i++)
	{
		randomSet.push_back(randomizer.Uniform(*(RANGE), *(RANGE + 1)));
	}

	return randomSet;
}

int Services::getGlobalBest(std::vector<Particle> swarm)
{
	std::vector<Particle>::iterator globalBest;
	globalBest = std::min_element(swarm.begin(), swarm.end(), [](Particle &a, Particle &b)
	{
		return a.getFitness() < b.getFitness();
	});

	return std::distance(swarm.begin(), globalBest);
}

int Services::getLessFit(std::vector<Particle> swarm)
{
    std::vector<Particle>::iterator lessFit;
    lessFit = std::max_element(swarm.begin(), swarm.end(), [](Particle &a, Particle &b)
    {
        return a.getFitness() < b.getFitness();
    });

    return std::distance(swarm.begin(), lessFit);
}

void Services::getGradient(Particle &particle, const float VMAX)
{
	std::vector<double> position = particle.getPosition();

	std::vector<double> g(position.size(), 0);
	double fx0 = evaluateParticle(position);
	double step = 1e-5;

	std::vector<double> xli;

	for (int i = 0, size = position.size(); i < size; i++)
	{
		xli = position;
		xli[i] = position[i] + step;
		g[i] = (evaluateParticle(xli) - fx0) / step;
	}

	g = trunc(g, VMAX);

	particle.setGradient(g);
}

std::vector<double> Services::trunc(std::vector<double> dataSet, const float VMAX)
{
	for (int i = 0, size = dataSet.size(); i < size; i++)
	{
		if (dataSet[i] < -VMAX)
		{
			dataSet[i] = -VMAX;
		}
		else if (dataSet[i] > VMAX)
		{
			dataSet[i] = VMAX;
		}
	}

	return dataSet;
}

void Services::updatePosition(Particle &particle)
{
	std::vector<double> position = particle.getPosition();
	std::vector<double> velocity = particle.getVelocity();

	for (int i = 0; i < position.size(); i++)
	{
		position[i] += velocity[i];
	}

	particle.setPosition(position);
}

void Services::truncSpace(Particle &particle, const float* RANGE)
{
	std::vector<double> position = particle.getPosition();

	for (int i = 0; i < position.size(); i++)
	{
		if (position[i] < RANGE[0])
		{
			position[i] = RANGE[0];
		}
		else if (position[i] > RANGE[1])
		{
			position[i] = RANGE[1];
		}
	}

	particle.setPosition(position);
}

void Services::updatePersonalBest(Particle &particle)
{
	double currentFitness = particle.getFitness();
	double personalBestFitness = particle.getPersonalBestFitness();
	std::vector<double> currentPosition = particle.getPosition();
	std::vector<double> personalBestPosition = particle.getPersonalBestPosition();

	if (currentFitness < personalBestFitness)
	{
		particle.setPersonalBestPosition(currentPosition);
		particle.setPersonalBestFitness(currentFitness);
	}
}

double Services::getDiversity(std::vector<Particle> swarm, double diagonalLength, const int swarmSize, const int dimensionSize)
{
	std::vector<double> average(dimensionSize, 0);
	double diversity = 0;
	std::vector<double> position;
	std::vector<double> diversityOperator(swarmSize, 0);

	for (int i = 0; i < swarmSize; i++)
	{
		position = swarm[i].getPosition();

		for (int j = 0; j < dimensionSize; j++)
		{
			average[j] += position[j];
		}
	}

	for (int i = 0; i < dimensionSize; i++)
	{
		average[i] /= swarmSize;
	}

	for (int i = 0; i < swarmSize; i++)
	{
		position = swarm[i].getPosition();

		for (int j = 0; j < dimensionSize; j++)
		{
			diversityOperator[i] += std::pow((position[j] - average[j]), 2);
		}
	}

	for (int i = 0; i < swarmSize; i++)
	{
		diversityOperator[i] = std::sqrt(diversityOperator[i]);

		diversity += diversityOperator[i];
	}

	return diversity / (diagonalLength * swarmSize);
}
