from typing import NewType
import numpy as np
import os
from collections import OrderedDict
import pprint

def mdp_to_dict(path):
  """
  inputs : takes in path to the mdp.
  returns : dictionary version of the txt file, with proper key, value pairs.
            transitions are under a single key.
            NOTE : everything is a string, do convert to float!

  """
  mdp_list = [a.strip('\n').split(' ') for a in open(path, 'r')]
  keys = [a[0] for a in mdp_list]
  keys = list(OrderedDict.fromkeys(keys))
  values = []
  transitions = []
  for i in mdp_list:
    if(i[0] == keys[3]):
      values.insert(3, i[1:])
    elif(i[0] == keys[4]):
      transitions.append(i[1:])
    else:
      values.append(i[-1])
  values.insert(4, transitions)

  data_value = dict(zip(keys, values))
  
  return data_value

