# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

#reading bandit instance from file\
path = '/home/kingshark1/Practise/RL_CS_747_IITB/assignment_1/cs747-pa1/instances/i-2.txt'
bandits_values = [a for a in open(path, 'r')]


# Define Action class
class Bandit:
  def __init__(self, m):
    self.m = m
    #self.horizon = horizon
    self.mean = 0
    self.N = 0
    #self.reg = horizon * 1
  
  # Choose a random action
  def choose(self): 
    reward = [0, 1]
    x = np.random.choice(reward, p=[1 - self.m, self.m])
    return x
  
  # Update the action-value estimate
  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * x
  
def eps_greedy(bandits, horizon, eps=0.02):
  """
  requires : 
  Bandit -  List with Bandit objects as elements already
            initiated with respective probability values.
  returns  :
  reg -     Expected cumulative Regret
  """
  #initializing empty storage container data
  max_expected_p = max([a.m for a in bandits])
  data = np.empty(horizon)
  for i in range(horizon):
    p = np.random.random()
    if p < eps:
      j = np.random.choice(len(bandits))
    else:
      j = np.argmax([a.mean for a in bandits])
    x = bandits[j].choose()
    bandits[j].update(x)
    
    #expected reward required for Regret
    data[i] = x
  #print(np.cumsum(data).shape)
  #print((max_expected_p * horizon))
  reg = (max_expected_p * horizon) - np.cumsum(data)
  reg = reg[-1]
  
  return reg

def run_experiment(m1, m2, m3, eps, N):
      
  bandits = [Bandit(m1),  Bandit(m2),  Bandit(m3)]
  
  data = np.empty(N)
  
  for i in range(N):
    # epsilon greedy
    p = np.random.random()
    if p < eps:
      j = np.random.choice(3)
    else:
      j = np.argmax([a.mean for a in  bandits])
    x =  bandits[j].choose()
    #print(x)
    bandits[j].update(x)
  
    # for the plot
    data[i] = x
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
  
  
  for a in  bandits:
    print(a.mean)
  
  return cumulative_average
  
  
if __name__ == '__main__':
  m1 = 0.4
  m2 = 0.3
  m3 = 0.5 
  m4 = 0.2
  m5 = 0.1
  bandits = [Bandit(m1),  Bandit(m2),  Bandit(m3), Bandit(m4), Bandit(m5)]
  
  horizon = 102400
  eps_greedy_reg = eps_greedy(bandits, horizon=horizon)
  print(f'Expected cumulative regret is : {eps_greedy_reg}')