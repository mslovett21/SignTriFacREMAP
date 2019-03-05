import time
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from os import getpid
from functools import partial
import ctypes as c


def matrix_multi(A,B,i,cols_p):
    
    print("I'm process", getpid())
    arr      = np.frombuffer(C_test.get_obj())
    arr      = arr.reshape(ans_rows.value, ans_cols.value) 
    res      = np.dot(A,np.transpose(B))
    arr[:,i] = res

    return 0



if __name__ == '__main__':
    
    A  = np.array([[1,-2,3],[0,2,1],[3,-4,0],[4,-2,1],[-3,0,2]])
    B  = np.array([[2,1,-3,1,0,1],[-4,2,0,-1,2,0], [2,-1,2,0,-1,0]])
    BT = np.transpose(B)
    
    C = np.dot(A,B)
    print("C")
    print(C)

    cpu_num      = mp.cpu_count()
    cpu_used     = 0
    num_cols     = BT.shape[0]
    # int values shared amongst all of the processes
    ans_rows     = mp.Value('i', A.shape[0])
    ans_cols     = mp.Value('i', B.shape[1])
    
    # 2d shared array that stores calculations of all of the procsses
    C_test  = mp.Array(c.c_double,A.shape[0]*B.shape[1])
    
    if (cpu_num > num_cols):
        cpu_used = num_cols
    else:
        cpu_used = cpu_num


    cols_p        = int(num_cols / cpu_used)
    last_used     = 0
    # number of columns of matrix B per process

    if(num_cols == (cols_p *cpu_used)):
        print("ALL GOOD")
    else:
        print("PROBLEM TO SOLVE SOON")

    for i in range(cpu_used):
        print(i)
        p = mp.Process(target = matrix_multi, args = (A,BT[i],i,cols_p))
        p.start()
        p.join()
    print("END OF STORY")
    print(C_test)
    C_test_b = np.frombuffer(C_test.get_obj())
    C_test   = C_test_b.reshape((ans_rows.value, ans_cols.value))
    print(C_test)

