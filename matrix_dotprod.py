import time
import os
import numpy as np
import sys





if __name__ == '__main__':

    start = time.time()
    #get the absolute path to the script
    script_dir = os.path.abspath(__file__) #<-- absolute dir the script is in
    script_dir = os.path.split(script_dir)[0] #i.e. /path/to/dir/
    
    abs_file_pathA = script_dir + str(sys.argv[1])
    abs_file_pathB = script_dir + str(sys.argv[2])
    A = np.loadtxt(open(abs_file_pathA, "rb"), delimiter = ",")
    B = np.loadtxt(open(abs_file_pathB, "rb"), delimiter = ",")
    
    C = np.dot(A,B)
    
    np.savetxt("dotAB.csv", C, delimiter = ",")
    
    end = time.time()
    print("Time used")
    print(end - start)
