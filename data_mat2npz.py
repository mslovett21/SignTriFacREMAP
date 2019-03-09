import numpy as np
from scipy import sparse
from math import pow
import sys
import os
import csv
from scipy.io import savemat, loadmat
from scipy import sparse



if __name__== '__main__':
    x = loadmat('orig_matlab_data/hetio_CdG.mat')
    data1 = x['hetio_CdG']
    data1 = data1.todense()
    data1 = data1*(-1)
    
    x = loadmat('orig_matlab_data/hetio_CuG.mat')
    data2 = x['hetio_CuG']
    data2 = data2.todense()
    data2 = data1 + data2
    data2_sparse = sparse.csr_matrix(data2)
    sparse.save_npz('data/hetio_CudG.npz',data2_sparse)
    
    x = loadmat('orig_matlab_data/hetio_DuG.mat')
    data1 = x['hetio_DuG']
    data1 = data1.todense()
    data1 = data1*(-1)
    
    x = loadmat('orig_matlab_data/hetio_DdG.mat')
    data2 = x['hetio_DdG']
    data2 = data2.todense()
    data2 = data1 + data2
    data2 = data2.transpose()
    data2_sparse = sparse.csr_matrix(data2)
    sparse.save_npz('data/hetio_DudG.npz',data2_sparse)
    
    x = loadmat('orig_matlab_data/hetio_CtD.mat')
    data = x['hetio_CtD']
    data = data.todense()
    data_sparse = sparse.csr_matrix(data)
    sparse.save_npz('data/hetio_CtD.npz',data_sparse)
    
    x = loadmat('orig_matlab_data/hetio_GiG.mat')
    data = x['hetio_GiG']
    data = data.todense()
    data_sparse = sparse.csr_matrix(data)
    sparse.save_npz('data/hetio_GiG.npz',data_sparse)
    
    x = loadmat('orig_matlab_data/hetio_DrD.mat')
    data = x['hetio_DrD']
    data = data.todense()
    data_sparse = sparse.csr_matrix(data)
    sparse.save_npz('data/hetio_DrD.npz',data_sparse)
    
    x = loadmat('orig_matlab_data/hetio_CsC.mat')
    data = x['hetio_CsC']
    data = data.todense()
    data_sparse = sparse.csr_matrix(data)
    sparse.save_npz('data/hetio_CsC.npz',data_sparse)
