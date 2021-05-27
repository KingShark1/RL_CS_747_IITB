import numpy as np
import argparse
import matplotlib.pyplot as plt
import tqdm

#initializing argument parser from command line
parser = argparse.ArgumentParser()

#adding optional arguments
parser.add_argument("-in", "--instance")
parser.add_argument("-al", "--algorithm")
parser.add_argument("-rs", "--randomSeed")
parser.add_argument("-ep", "--epsilon")
parser.add_argument("-hz", "--horizon")
args = parser.parse_args()

#sets global constants taken from command line
if args.instance:
  inst = args.instance
if args.algorithm:
  algo = args.algorithm
if args.randomSeed:
  r_seed = args.randomSeed
if args.epsilon:
  eps = args.epsilon
if args.horizon:
  horizon = args.horizon

#initializes the random generator to given seed
np.random.seed(r_seed)

# Define Action class
class Actions:
  def __init__(self, m):
    self.m = m
    self.mean = 0
    self.N = 0
  
  # Choose a random action
  def choose(self): 
    reward = [0, 1]
    x = np.random.choice(reward, p=[1 - self.m, self.m])
    return x
  
  # Update the action-value estimate
  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * x


#simulates epsilon greedy algorithm
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
  """
  #plotting
  plt.plot(np.cumsum(rew)/(np.arange(horizon) + 1))
  plt.plot(np.ones(102400)*bandits[0].m)
  plt.xscale('log')

  plt.show()
  """
  
  reg = reg[-1]
  return reg

#simulates UCB algorithm with KL bound
def KL_UCB(eps=0.1):
  """
  """
  return 0

#simulates thompson sampling algorithm
def thompson_sampling(eps=0.1):
  """
  """
  return 0

# simulates thomson sampling algorithm with given
# given hint about the prior on arms
def th_samp_with_hint(eps=0.1):
  """
  """
  return 0