import numpy as np
import scipy as sp
from scipy import sparse
import sys







if __name__=='__main__':

    A = sys.argv[1]
    A = sparse.load_npz(A)
    print(A.get_shape())
