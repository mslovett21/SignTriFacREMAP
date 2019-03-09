import time
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from os import getpid
from functools import partial
import ctypes as c
import sys
from scipy import sparse
import csv


def matrix_multi(A,B,last_used,cols_per_proc):
    
    arr      = np.frombuffer(C_test.get_obj())
    arr      = arr.reshape(ans_rows.value, ans_cols.value)
    res      = np.dot(A,np.transpose(B))
    arr[:,last_used:(last_used+cols_per_proc)] = res

    return 0



if __name__ == '__main__':

    
    #get the absolute path to the script
    script_dir = os.path.abspath(__file__) #<-- absolute dir the script is in
    script_dir = os.path.split(script_dir)[0] #i.e. /path/to/dir/
    
    abs_file_pathA = script_dir +"/"+ str(sys.argv[1])
    print("abs path A")
    print(abs_file_pathA)
    abs_file_pathB = script_dir + "/"+ str(sys.argv[2])
    print("abs path B")
    print(abs_file_pathB)
    A = np.loadtxt(open(abs_file_pathA, "rb"),delimiter = ",")
    A = sparse.csr_matrix(A)
    print("Read in A")
    B = np.loadtxt(open(abs_file_pathB,"rb"),delimiter = ",")
    print("Read in B")
    B = sparse.csr_matrix(B)
    
    start = time.time()
    BT = np.transpose(B)
    
    A = A.todense()
    BT = BT.todense()
    cpu_num      = mp.cpu_count()
    cpu_used     = 0
    num_cols     = BT.shape[0]
    # int values shared amongst all of the processes
    ans_rows     = mp.Value('i', A.shape[0])
    ans_cols     = mp.Value('i', B.shape[1])
    
    # 2d shared array that stores calculations of all of the procsses
    C_test  = mp.Array(c.c_double,A.shape[0]*B.shape[1])
    
    if (cpu_num > num_cols):
        cpu_used  = num_cols
    else:
        cpu_used  = cpu_num
    cpu_used = 10
    cols_per_proc = int(num_cols / cpu_used)
    last_used     = 0
    
    # number of columns of matrix B per process
    print("DATA ABOUT THE PROCEDURE:")
    print("Number of columns in B: %d" % (num_cols))
    print("Number of CPU used: %d" %(cpu_used))
    print("Number of columns per CPU: %d" % (cols_per_proc))


    for i in range(cpu_used):
        p = mp.Process(target = matrix_multi, args = (A,BT[last_used:(last_used+cols_per_proc)],last_used,cols_per_proc))
        p.start()
        last_used = last_used+cols_per_proc
        p.join()

    if(num_cols == (cols_per_proc*cpu_used)):
        pass
    else:
        print("Running additional process with a reminder...")
        difference = num_cols - last_used
        print("reminder %d" %(difference))
        p = mp.Process(target = matrix_multi, args = (A,BT[last_used:(last_used+difference)],last_used, difference))
        p.start()
        p.join()
    
    C_test_b = np.frombuffer(C_test.get_obj())
    C_test   = C_test_b.reshape((ans_rows.value, ans_cols.value))
    np.savetxt("mpAB.csv", C_test, delimiter = ",")
    
    end = time.time()
    print("Time used")
    print(end - start)
