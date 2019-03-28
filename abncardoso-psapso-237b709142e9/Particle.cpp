#include "Particle.h"

std::vector<double> Particle::getPosition()
{
	return position;
};

void Particle::setPosition(std::vector<double> position)
{
	this->position = position;
}

void Particle::updatePosition(std::vector<double> velocity)
{
	for (int i = 0; i < position.size(); i++)
	{
		position[i] += velocity[i];
	}
}

std::vector<double> Particle::getVelocity()
{
	return velocity;
};

void Particle::setVelocity(std::vector<double> velocity)
{
	this->velocity = velocity;
}

double Particle::getFitness()
{
	return fitness;
}

void Particle::setFitness(double fitness)
{
	this->fitness = fitness;
};

std::vector<double> Particle::getGradient()
{
	return gradient;
};

void Particle::setGradient(std::vector<double> gradient)
{
	this->gradient = gradient;
}

std::vector<double> Particle::getPersonalBestPosition()
{
	return personalBestPosition;
}

void Particle::setPersonalBestPosition(std::vector<double> newPersonalBest)
{
	personalBestPosition = newPersonalBest;
}

double Particle::getPersonalBestFitness()
{
	return personalBestFitness;
}

void Particle::setPersonalBestFitness(double newPersonalBest)
{
	personalBestFitness = newPersonalBest;
}

double Particle::getOldFitness()
{
	return oldFitness;
}

void Particle::setOldFitness(double newValue)
{
	oldFitness = newValue;
}