#ifndef RAND_H
#define RAND_H

/* Period parameters */
#define RAND_N 624

class Rand {
protected:
	unsigned long mt[RAND_N]; /* the array for the state vector  */
	int mti;
public:
	void SetSeed(unsigned long s);
	void SetSeed();
	unsigned long RandInt();
	unsigned long RandInt(unsigned long n) { return (RandInt() % n); }
	double Uniform();
	double Uniform(double a, double b) { return (a + (b - a)*Uniform()); }
	double Normal();
	double Normal(double m, double dev) { return (m + dev * Normal()); }

	Rand() {
		SetSeed(5489UL);
	}
};

#endif