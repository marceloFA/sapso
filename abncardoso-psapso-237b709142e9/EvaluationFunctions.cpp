#include "EvaluationFunctions.h"

#define PI 2.1416

double EvaluationFunctions::sphere(std::vector<double> x)
{
	double y = 0;

	for (int i = 0; i < x.size(); i++)
	{
		y += x[i] * x[i];
	}
	return y;
}

double EvaluationFunctions::rosenbrock(std::vector<double> x)
{
	int a = 1;
	int b = 100;
	double y = 0;
	double xplus1;
	double xminus1;
	double xminus1s;
	double xaux1;
	double xaux2;
	for (int i = 0, size = x.size(); i < size - 1; i++)
	{
		xplus1 = x[i + 1];
		xminus1 = x[i];
		xminus1s = std::pow(xminus1, 2);
		xaux1 = xplus1 - xminus1s;
		xaux2 = (a - xminus1);

		y += b * (xaux1 * xaux1) + (xaux2 * xaux2);
	}

	return y;
}

double EvaluationFunctions::rastrigin(std::vector<double> x)
{
	int a = 10;
	int n = x.size();
	double y = a*n;

	for (int i = 0; i < n; i++)
	{
		y += (x[i] * x[i]) - (a*std::cos(2*PI*x[i]));
	}

	return y;
}

double EvaluationFunctions::griewank(std::vector<double> x)
{
	double comp1 = 0;
	double comp2 = 1;
	double y = 1;
	
	for (int i = 0, size = x.size(); i < size; i++)
	{
		comp1 += (x[i] * x[i])/4000;
		comp2 *= std::cos(x[i] / std::sqrt(i + 1));
	}

	y += comp1 - comp2;

	return y;
}

double EvaluationFunctions::ackley(std::vector<double> x)
{
	int a = 20;
	float b = 0.2f;
	float c = 2 * PI;
	double comp1 = 0;
	double comp2 = 0;
	double y = 0;

	for (int i = 0, size = x.size(); i < size; i++)
	{
		comp1 += x[i] * x[i];
		comp2 += std::cos(2*PI*x[i]);
	}

	y = (-a)*exp((-b)*std::sqrt(comp1 / x.size())) -
		exp(comp2/x.size()) + a + exp(1);

	return y;
}

double EvaluationFunctions::ellipsoid(std::vector<double> x)
{
	double y = 0;

	for (int i = 0, size = x.size(); i < size; i++)
	{
		y += (i + 1) * x[i] * x[i];
	}

	return y;
}

double EvaluationFunctions::schaffer2(std::vector<double> x)
{
	double x1 = std::pow(x[0], 2);
	double x2 = std::pow(x[1], 2);
	double numerator = std::pow(std::sin(x1 - x2), 2) - 0.5;
	double denominator = std::pow((1 + 0.001 * (x1 + x2)), 2);
	double y = 0;

	y = 0.5 + (numerator / denominator);

	return y;
}

double EvaluationFunctions::alpine(std::vector<double> x)
{
	double y = 0;

	for (int i = 0, size = x.size(); i < size; i++)
	{
		y += std::abs(x[i] * std::sin(x[i]) + 0.1 * x[i]);
	}

	return y;
}

double EvaluationFunctions::levi(std::vector<double> x)
{
	int size = x.size();
	std::vector<double> w;
	double y = 0;

	for (int i = 0; i < size; i++)
	{
		w.push_back(1 + (x[i] - 1) / 4);
	}

	double term1 = std::pow(std::sin(PI * w[0]), 2);

	double term2 = 0;

	for (int i = 0; i < size - 1; i++)
	{
		term2 += std::pow((w[i] - 1), 2) * (1 + 10 * std::pow(std::sin(PI * w[i] + 1), 2));
	}

	double term3 = std::pow(w[size-1], 2) * (1 + std::pow(std::sin(2 * PI * w[size-1]), 2));

	y = term1 + term2 + term3;

	return y;
}

double EvaluationFunctions::levi13(std::vector<double> x)
{
	double x1 = x[0];
	double x2 = x[1];
	double y = 0;

	y = std::pow(std::sin(3 * PI * x1), 2);

	y += std::pow((x1 - 1), 2) * (1 + std::pow(std::sin(3 * PI * x2), 2));

	y += std::pow((x2 - 1), 2) * (1 + std::pow(std::sin(2 * PI * x2), 2));

	return y;
}