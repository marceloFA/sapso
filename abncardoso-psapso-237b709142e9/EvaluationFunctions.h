#pragma once

#include <vector>
#include <cmath>

class EvaluationFunctions
{
	friend class Services;
public:
	static double sphere(std::vector<double> x); // [-100, 100]
	static double rosenbrock(std::vector<double> x); // [-5.12, 5.12]
	static double rastrigin(std::vector<double> x); // [-5.12, 5.12]
	static double griewank(std::vector<double> x); // [-100, 100]
	static double ackley(std::vector<double> x); // [-50, 50] ------------------> Verificar
	static double ellipsoid(std::vector<double> x); // [-5.12, 5.12]
	static double schaffer2(std::vector<double> x); // [-100, 100]
	static double alpine(std::vector<double> x); // [0, 20]
	static double levi(std::vector<double> x); // [-10, 10] ---------------------> Errado
	static double levi13(std::vector<double> x); // [-10, 10] --------------------> Errado
};