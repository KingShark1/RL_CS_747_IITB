# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
  def pull_arm(self): 
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
  #initializing empty storage container rew
  max_expected_p = max([a.m for a in bandits])
  rew = np.empty(horizon)
  for i in tqdm(range(horizon)):
    p = np.random.random()
    if p < eps:
      j = np.random.choice(len(bandits))
    else:
      j = np.argmax([a.mean for a in bandits])
    x = bandits[j].pull_arm()
    bandits[j].update(x)
    
    #expected reward required for Regret
    rew[i] = x
  #print(np.cumsum(rew).shape)
  #print((max_expected_p * horizon))
  reg = (max_expected_p * horizon) - np.cumsum(rew)
  reg = reg[-1]
  
  return reg

#simulates Upper Confidence Bound algorithm
def UCB(bandits, horizon, eps=0.1):
  """
  requires : 
  Bandit -  List with Bandit objects as elements already
            initiated with respective probability values.
  returns  :
  reg -     Expected cumulative Regret
  
  Simulates the Upper Confidence Bound algorithm
    - at time t define 
      UCB_of_arm_a_at_t = mean_of_arm_a + (2*log(t)/N)**0.5
      N :: defined as number of times arm a has been selected
            prior to t
    - sample an arm for which UCb_of_a_at_t is maximum
  """
  #initialize sampling so bandits.N is at least 1 for every arm
  for bandit in bandits:
    x = bandit.pull_arm()
    bandit.update(x)

  # helper function for ucb
  def calculate_ucb_at_t(bandits, t):
    updated_ucb = np.empty(len([a.m for a in bandits]))
    for j in range(len(bandits)):
      updated_ucb[j] = bandits[j].mean + np.sqrt((2*np.log(t)/bandits[j].N))
        
    return np.array(updated_ucb)

  max_expected_p = max([a.m for a in bandits])
  rew = np.empty(horizon)
  #ucb_at_t = np.empty(len([a.m for a in bandits]))
  for t in tqdm(range(1, horizon)):
    # calculates ucb
    ucb_at_t = calculate_ucb_at_t(bandits, t)
    #print(ucb_at_t)
    j = np.argmax(ucb_at_t)
    x = bandits[j].pull_arm()
    bandits[j].update(x)
    
    rew[t] = x
  
  reg = (max_expected_p*horizon - np.cumsum(rew))
  
  #plotting
  plt.plot(np.cumsum(rew)/(np.arange(horizon) + 1))
  plt.plot(np.ones(102400)*bandits[0].m)
  plt.xscale('log')

  plt.show()
  
  #print(rew[0:10])
  reg = reg[-1]
  return reg
  
if __name__ == '__main__':
  m1 = 0.4
  m2 = 0.3
  m3 = 0.5
  m4 = 0.2
  m5 = 0.1
  bandits = [Bandit(m1),  Bandit(m2),  Bandit(m3), Bandit(m4), Bandit(m5)]
  
  horizon = 102400
  #print(ucb_at_t)
  #eps_greedy_reg = eps_greedy(bandits, horizon=horizon)
  #print(f'Expected cumulative regret is : {eps_greedy_reg}')

  ucb_reg = UCB(bandits, horizon=horizon)
  print(f'Expected cumulative regret for UCB is : {ucb_reg}')

  
  