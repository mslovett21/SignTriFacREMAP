import time
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from os import getpid
from functools import partial
import ctypes as c
import sys

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
    
    abs_file_pathA = script_dir + str(sys.argv[1])
    abs_file_pathB = script_dir + str(sys.argv[2])
    A = np.loadtxt(open(abs_file_pathA, "rb"), delimiter = ",")
    B = np.loadtxt(open(abs_file_pathB, "rb"), delimiter = ",")
    
    BT = np.transpose(B)
    
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
        p = mp.Process(target = matrix_multi, args = (A,BT[last_used:(last_used+difference)],last_used, difference))
        p.start()
        p.join()

    print("RESULT")
    
    C_test_b = np.frombuffer(C_test.get_obj())
    C_test   = C_test_b.reshape((ans_rows.value, ans_cols.value))
    np.savetxt("mpAB.csv", C_test, delimiter = ",")
    
    if (C[2][34] == C_test[2][34]):
        print("ALL SAME")
    else:
        print("DIFFERENT")
