import numpy as np
import sys
import os
import csv
from sklearn.model_selection import KFold
from scipy import sparse




def mask_the_values(matrix_copy,chosen_mask,all_indices,j):
	masked_values = []
	index_file = open('ten_fold/index_fold_' + str(j) + '.csv','a')
	for i in range(len(chosen_mask)):
		masked_values.append(matrix_copy[all_indices[chosen_mask[i]][0]][all_indices[chosen_mask[i]][1]])
		matrix_copy[all_indices[chosen_mask[i]][0]][all_indices[chosen_mask[i]][1]] = 0.0
		index_file.write(str(all_indices[chosen_mask[i]][0]))
		index_file.write(',')
		index_file.write(str(all_indices[chosen_mask[i]][1]))	
		index_file.write('\n')
	index_file.close()
	return masked_values, matrix_copy

if __name__== '__main__':
	i=1

	# read in the file with the matrix
	print(sys.argv[0])
	D_matrix_fh = sys.argv[1]
	D_matrix    = np.loadtxt(open(D_matrix_fh, "rb"), delimiter=",")


	matrix_shape_x,matrix_shape_y =  D_matrix.shape

	#gather all valid pairs of indexes in the matrix
	upper_trian_indices           =  np.triu_indices(matrix_shape_x, k=0,  m=matrix_shape_y)
	lower_trian_indices           =  np.tril_indices(matrix_shape_x, k=-1, m=matrix_shape_y)
	lower_indexes                 =  np.array(lower_trian_indices).transpose()
	upper_indexes                 =  np.array(upper_trian_indices).transpose()
	all_indices                   =  np.concatenate((upper_indexes,lower_indexes), axis= 0)
	kf                            =  KFold(n_splits=10,shuffle=True)
   
	# each time mask different 10% of the values, store the values that are masked                     
	for notchosen_index, chosen_index in kf.split(all_indices):
            chosen_mask                       = chosen_index
            print("Chosen index length:")
            print(len(chosen_index))
            masked_values, D_masked           = mask_the_values(D_matrix, chosen_mask,all_indices,i)
            mask_file = open('ten_fold/masked_val_fold_' + str(i) + '.csv','a')
            for j in range(len(masked_values)):
                mask_file.write(str(masked_values[j]))
                mask_file.write("\n")
            mask_file.close()
            D_masked_sparse = sparse.csr_matrix(D_masked)
            sparse.save_npz('ten_fold/matrix_fold_' + str(i) + '.npz',D_masked_sparse)
            print("Finished matrix %d" % (i))
            i = i+1
