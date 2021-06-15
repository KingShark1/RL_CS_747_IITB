import argparse, random, os
parser = argparse.ArgumentParser()
import numpy as np
from assignment_2 import utils
random.seed(0)

def value_iteration(path_to_mdp):
  mdp = utils.mdp_to_dict(path_to_mdp)

def howard_policy_iteration(path_to_mdo):
  mdp = utils.mdp_to_dict(path_to_mdo)

def linear_programming(path_to_mdp):
  mdp = utils.mdp_to_dict(path_to_mdp)

if __name__ == "__main__":
  parser.add_argument("--mdp", type=str)
  parser.add_argument("--algorithm", type=str, default="hpi")
  args = parser.parse_args()

  