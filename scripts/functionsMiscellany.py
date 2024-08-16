# miscellaneous functions for the vertebral constraint analysis

#%% libraries

from skbio import TreeNode
from io import StringIO
import pandas as pd
import numpy as np
import sympy as sp

#%% simple function to simplify constraints

def simplifyConstraints(x):

    # change the sign of the constraint vectors so that the first non-zero element is positive
    for i in range(len(x)):
        indNonZero = np.where(x[i,:]!=0)[0][0]
        if x[i,indNonZero]<0:
            x[i,:] = -x[i,:]
            
    # make sure we reduce/simplify the constraint vectors as much as possible
    for i in range(len(x)):
        x[i,:] = x[i,:]/np.min(np.abs(x[i,x[i,:]!=0]))
    
    x += 0.
    
    return x
            