import numpy as np

GiG_fh  = "hetio_GiG_org.txt"
CudG_fh = "hetio_CudG_org.txt"
DudG_fh = "hetio_DudG_org.txt"

GiG_matrix = np.loadtxt(open(GiG_fh,"rb"), delimiter=",")
GiG_matrix = GiG_matrix[0:2000,0:2000]
np.savetxt("hetio_GiG.txt", GiG_matrix, delimiter=",")


CudG_matrix = np.loadtxt(open(CudG_fh,"rb"), delimiter=",")
CudG_matrix = CudG_matrix[:,0:2000]
np.savetxt("hetio_CudG.txt", CudG_matrix, delimiter =",")

#this is the file with transposed matrix it is GD
DudG_matrix = np.loadtxt(open(DudG_fh),delimiter=",")
DudG_matrix = DudG_matrix[0:2000,:]
np.savetxt("hetio_DudG.txt",DudG_matrix, delimiter=",")


