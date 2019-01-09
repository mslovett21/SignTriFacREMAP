import numpy as np
from scipy import sparse
from math import pow
import sys
import os
import csv
from scipy.io import savemat, loadmat




if __name__== '__main__':
	
	x = loadmat('hetio_CdG.mat')
	data1 = x['hetio_CdG']
	data1 = data1.todense()
	data1 = data1*(-1)
	
	x = loadmat('hetio_CuG.mat')
	data2 = x['hetio_CuG']
	data2 = data2.todense()
	data2 = data1 + data2
	np.savetxt('data/hetio_CudG.txt',data2,delimiter=',')

	x = loadmat('hetio_DuG.mat')
	data1 = x['hetio_DuG']
	data1 = data1.todense()
	data1 = data1*(-1)

	x = loadmat('hetio_DdG.mat')
	data2 = x['hetio_DdG']
	data2 = data2.todense()
	data2 = data1 + data2
	np.savetxt('data/hetio_DudG.txt',data2,delimiter=',')

	x = loadmat('hetio_CtD.mat')
	data = x['hetio_CtD']
	data = data.todense()
	np.savetxt('data/hetio_CtD.txt',data,delimiter=',')

	x = loadmat('hetio_GiG.mat')
	data = x['hetio_GiG']
	data = data.todense()
	np.savetxt('data/hetio_GiG.txt',data,delimiter=',')

	x = loadmat('hetio_DrD.mat')
	data = x['hetio_DrD']
	data = data.todense()
	np.savetxt('data/hetio_DrD.txt',data,delimiter=',')

	x = loadmat('hetio_CsC.mat')
	data = x['hetio_CsC']
	data = data.todense()
	np.savetxt('data/hetio_CsC.txt',data,delimiter=',')