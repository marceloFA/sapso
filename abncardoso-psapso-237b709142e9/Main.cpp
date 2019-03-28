#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits.h>

#include "omp.h"

#include "SAPSOServices.h"
#include "Particle.h"
#include "Rand.h"

int main(int argc, char* argv[])
{

    ////////////////// Test Function //////////////////
    int DIM = 30; // Number of dimensions.
    float RANGE[2] = { -5.12, 5.12 };  // Search space.
    std::string functionName("sphere");


    ////////////////// PSO Parameters //////////////////
    const int NPART = 20; // Number of particles.
    const int MAXITER = 5000; // Max number of iterations.
    const double STOPC = 1e-10; // Stop criteria.
    int k = 1;
    int CMAX = 3; // Number of consecutive evaluations.
    float SC = 2; //Social coeficient.
    double GC = 1e-3; //Gradient coeficient
    double DT[2] = { 1e-6, .25 }; // dlow and dhigh.

    const float IWMIN = .4f; // Minimum inertia weight
    const float IWMAX = .7f; // Maximum inertia weight
    double diagonalLength = 0; // Diagonal length of the search space.

    const int threadNum = 5;

    for(int i = 0; i < DIM; i++)
    {
        diagonalLength += std::pow(RANGE[1] - RANGE[0], 2);
    }
    diagonalLength = std::sqrt(diagonalLength);

    const float VMAX = k * (RANGE[1] - RANGE[0]) / 2; // Maximum velocity.
    const float VMAXSet[2] = {-VMAX, VMAX};

    std::vector<float> IW(MAXITER);

    //Iteration dependant inertial weight.
    for (int i = 0; i < MAXITER; i++)
    {
        IW[i] = IWMAX - (i + 1) * (IWMAX - IWMIN) / MAXITER;
    }

    //Setting PRNG's seed.
    std::vector<int> seed;
    Rand randomizer;
    randomizer.SetSeed(time(NULL));

    for(int i = 0; i < threadNum; i++)
    {
        seed.push_back(randomizer.RandInt(INT_MAX));
    }

    Particle *sharedParticles[threadNum];
    Particle swarmBest;


    //Coleta de dados
    std::string functions[8] = {"sphere", "rosenbrock", "rastrigin", "griewank", "ackley", "ellipsoid", "schaffer2", "alpine"};
    int SCs[8] = { 2, 2, 3, 3, 4, 2, 2, 4 };
    double DTs[8] = {1e-6, 1e-6, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2 };
    double GCs[8] = { 1e-2, 1e-3, 1e-3, 1e-2, 1e-1, 1e-2, 1e-2, 1e-2 };
    int DIMs[3] = {10, 20, 30};
    float RANGEs[8] = { 100, 5.12, 5.12, 100, 47.5, 5.12, 100, 20 };

    std::ofstream outputFile;
    outputFile.open("Results.csv");
    outputFile << "Function; dimensions; avg. Iteration; avg Fitness\n";

    for (int k = 0; k < 8; k++)
    {
        SC = SCs[k];
        DT[0] = DTs[k];
        GC = GCs[k];
        functionName = functions[k];
        RANGE[0] = -RANGEs[k];
        RANGE[1] = RANGEs[k];

        if (k == 7)
            RANGE[0] = 0;

        for (int l = 0; l < 3; l++)
        {
            double avg = 0;
            int avgIter = 0;

            DIM = DIMs[l];
            for(int m = 0; m < 20; m++)
            {
                int iter = 0;
                bool solutionFound = false;
#pragma omp parallel num_threads(threadNum)
                {
                    int dir = 1;
                    int globalBest = 0;
                    double diversity = 0;
                    std::vector<int> particleDecision(NPART, 0); // Decision of every particle.
                    std::vector<int> stagnationCounter(NPART, 0); // Saves the number of stagnant generations.
                    SAPSOServices services(functionName, seed[omp_get_thread_num()]);
                    std::vector<Particle> swarm;

                    //Swarm and service initialization.
                    for (int i = 0; i < NPART; i++)
                    {
                        Particle particle;

                        particle.setPosition(services.generateRandomSet(DIM, RANGE));
                        particle.setVelocity(services.generateRandomSet(DIM, VMAXSet));
                        particle.setFitness(services.evaluateParticle(particle.getPosition()));
                        particle.setPersonalBestPosition(particle.getPosition());
                        particle.setPersonalBestFitness(particle.getFitness());
                        particle.setOldFitness(particle.getFitness());

                        swarm.push_back(particle);
                    }

                    globalBest = services.getGlobalBest(swarm);

                    //Main loop.
                    for (int i = 0; i < MAXITER; i++)
                    {
                        for (int j = 0; j < NPART; j++)
                        {
                            services.getGradient(swarm[j], VMAX);

                            services.updateVelocity(swarm[j], swarm[globalBest].getPersonalBestPosition(), particleDecision[j], IW[i], SC, GC, dir, VMAX);

                            services.updatePosition(swarm[j]);
                            services.truncSpaceSAPSO(swarm[j], particleDecision[j], stagnationCounter[j], RANGE);

                            swarm[j].setFitness(services.evaluateParticle(swarm[j].getPosition()));

                            services.updatePersonalBest(swarm[j]);

                            if (swarm[j].getPersonalBestFitness() < swarm[globalBest].getPersonalBestFitness())
                                globalBest = j;
                        }

                        services.updateImportance(swarm, particleDecision, stagnationCounter, CMAX, globalBest);

                        diversity = services.getDiversity(swarm, diagonalLength, NPART, DIM);

                        services.updateDir(dir, diversity, particleDecision, DT, NPART);

                        for (int j = 0, size = swarm.size(); j < size; j++)
                        {
                            swarm[j].setOldFitness(swarm[j].getFitness());
                        }

                        //            std::cout << "Iteration: " << i << "\n Best Fitness: " << swarm[globalBest].getPersonalBestFitness() << "\n Diversity: " << diversity << "\n dir: " << dir << "\n\n";

                        //                        if (swarm[globalBest].getPersonalBestFitness() < STOPC)
                        //                        {
                        //                            solutionFound = true;
                        //                        }

#pragma omp single
                        iter++;

                        //#pragma omp barrier
                        //                        if(solutionFound)
                        //                        {
                        //                            break;
                        //                        }

                        if( (i + 1) % 500 == 0)
                        {
                            sharedParticles[omp_get_thread_num()] = &swarm[globalBest];
#pragma omp barrier
#pragma omp master
                            {
                                int best = 0;

                                for(int n = 1; n < threadNum; n++)
                                {
                                    if(sharedParticles[best]->getFitness() > sharedParticles[n]->getFitness())
                                        best = n;
                                }

                                swarmBest = *sharedParticles[best];
                            }
#pragma omp barrier
                            if(omp_get_thread_num() != 0)
                            {
                                swarm[randomizer.RandInt(threadNum)] = swarmBest;
                                globalBest = services.getGlobalBest(swarm);
                            }
                        }
                    }

#pragma omp master
                    {
                        avg += swarmBest.getPersonalBestFitness();
                        std::cout << "Execution: " << m << ": \n";
                        std::cout << "Function: " << functionName << ": \n";
                        std::cout << "Dimension: " << DIM << ": \n";
                        std::cout << omp_get_thread_num() << ": \n";
                        std::cout << "Best Fitness: " << swarmBest.getPersonalBestFitness() << "\n Diversity: " << diversity << "\n dir: " << dir << "\n\n";

                        std::cout << "X = [ ";
                        for(int aux = 0, size = swarmBest.getPersonalBestPosition().size(); aux < size; aux++)
                        {
                            std::cout << swarmBest.getPersonalBestPosition()[aux] << " ";
                        }
                        std::cout << " ]\n\n";
                    }
                }
                avgIter += iter;
            }
            avgIter /= 20;
            avg /= 20;
            outputFile << functionName << "; " << DIM << ", " << avgIter << ", " << avg << "\n";
            printf("avg: %E\n", avg);
            printf("avg. iter: %d\n", avgIter);
        }
    }
    outputFile.close();
}
