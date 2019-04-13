import numpy as np
class TestFunctions():
  '''
    Test functions for single-objective optimization
  '''

  #Global Minimum:
  rastrigin_min = np.array([0.,0.])
  ackley_min = np.array([0.,0.])
  sphere_min = np.array([0.,0.])
  rosenbrock_min = np.array([1.,1.])
  beale_min = np.array([3.,0.5])
  gold_min = np.array([0.,-1.])
  booth_min = np.array([1.,3.])
  bukin_min = np.array([-10.,1.])
  matyas_min = np.array([0.,0.])
  levi_13_min = np.array([1.,1.])
  himmelblau_min = np.array([3.,2.])

  #Search Space:
  rastrigin_space = [-5.12,5.12]
  ackley_space = [-5.,5.]
  sphere_space = [-10.,10.]
  rosenbrock_space = [-10.,10.]
  beale_space = [-4.5,4.5]
  gold_space = [-2.,2.]
  booth_space = [-10.,10.]
  bukin_space = [-15.,3.]
  matyas_space = [-10.,10.]
  levi_13_space = [-10.,10.]
  himmelblau_space = [-5.,5.]

  
  def rosenbrock(self,x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


  def sphere(self,x):
    """The Sphere function"""
    return sum(i*i for i in x)


  def rastrigin(self,x):
    """The Rastrigin function"""
    return 10 * len(x) + sum(i * i - 10 * np.cos(2 * np.pi * i) for i in x)


  def himmelblau(self,x):
    """The Himmelblau function"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


  def ackley(self,x):
    """The Ackley function"""
    dim = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(dim):
        sum1 += x[i]**2
        sum2 += np.cos(2*np.pi*x[i])

    return -20*np.exp(-0.2*np.sqrt(sum1/dim)) - np.exp(sum2/dim) + 20 + np.e


  def levi_13(self,x):
    """The Levy Function N. 13"""
    return np.sin(3*np.pi*x[0])**2 + (x[0] - 1)**2*(1 + np.sin(3*np.pi*x[1])**2) + (x[1] - 1)**2*(1 + np.sin(2*np.pi*x[1])**2)


  def matyas(self,x):
    """The Matyas function"""
    return (0.26 * (x[0]**2 + x[1]**2)) - (0.48 * x[0] * x[1])


  def booth(self,x):
    """The Booth function"""
    return ((x[0] + 2*x[1] - 7)**2) + ((2*x[0] + x[1] +-5)**2)


  def gold(self, x):
    '''The Goldstein-Price function'''
    return (1+((x[0]+x[1]+1)**2) * (19-14*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+(3*x[1]**2))) + ((30+(2*x[0]-3*x[1])**2) * (18-32*x[0]+(12*x[0]**2)+48*x[1]-36*x[0]*x[1]+(27*x[1]**2)))


  def beale(self,x):
    """ The Beale function """
    return (1.5-x[0]+(x[0]*x[1]))**2 + (2.25-x[0]*x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2


  def bukin(self, x):
    '''The Bukin N.6 function'''
    return 100*np.sqrt(x[1]-0.001*x[0]**2) + 0.001 * abs(x[0]+10)