import argparse, random, os
parser = argparse.ArgumentParser()
import numpy as np
import utils
random.seed(0)

def value_iteration(path_to_mdp):
  mdp = utils.mdp_to_dict(path_to_mdp)
  '''
  Computes optimal MDP policy and its value via Value Iteration Method.

  Inputs - path to the mdp
  Outputs - pi(state), val_function(s)
  '''
  #real array sequence of value functions
  val_func = [1]
  
  #action array pi
  pi = []

  theta = 0.1
  k =  0
  s = np.asarray([1 for a in np.ones(mdp['numStates'])])
  
  # v_k_s is list of values computed at step k for all states s
  #val function is list of optimal value function at steps k
  while(a>theta for a in s):
    k += 1
    v_k_s = s
    for i in range(len(s)):
      v_k_s[i] = mdp['transition'][][-1](mdp['transition'][][-2] + mdp['discount']*val_func[k-1])
      s[i] = v_k_s[i] - v_k_s[]

    val_func.append(max(v_k_s))
          
  for i in range(len(s)):
    pi[i] = 
  return pi, val_func

def howard_policy_iteration(path_to_mdo):
  mdp = utils.mdp_to_dict(path_to_mdo)

def linear_programming(path_to_mdp):
  mdp = utils.mdp_to_dict(path_to_mdp)

if __name__ == "__main__":
  parser.add_argument("--mdp", type=str)
  parser.add_argument("--algorithm", type=str, default="hpi")
  args = parser.parse_args()

