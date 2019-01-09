Expected data format

to run the toy dataset:

python signTriFacREMAP.py g_matrix.txt d_matrices.txt a_matrices.txt w_matrix.txt 5 0.89 0.34

#where
#Expected arguments: g_matrix.txt d_matrix.txt a_matrix.txt w_matrix.txt max_iteration alpha beta [optional] b_matrix.txt
#if b_matrix.txt is not provided b=0.5 for each signed relation
#Results of the computations are saved in /F_results and /P_results -each matrix in separate file

#g_matrix.txt
0,2,1,1,0
0,0,1,2,2
0,0,0,2,2
0,0,0,0,0
0,0,0,0,0

#a_matrix.txt
/A_matrices/layer00.txt,0,0,0,0
0,/A_matrices/layer11.txt,0,0,0
0,0,/A_matrices/layer22.txt,0,0
0,0,0,/A_matrices/layer33.txt,0
0,0,0,0,/A_matrices/layer44.txt

#d_matrices.txt
0,/D_matrices/relation01.txt,/D_matrices/relation02.txt,/D_matrices/relation03.txt,0
0,0,/D_matrices/relation12.txt,/D_matrices/relation13.txt,/D_matrices/relation14.txt
0,0,0,/D_matrices/relation23.txt,/D_matrices/relation24.txt
0,0,0,0,0
0,0,0,0,0

#w_matrix.txt
0,0.23,0.5,0.78,0
0,0,0.45,0.55,0.34
0,0,0,0.34,0.39
0,0,0,0,0
0,0,0,0,0

#b_matrix.txt
0,0.25,0,0,0
0,0,0,0.32,0.78
0,0,0,0.11,0.24
0,0,0,0,0
0,0,0,0,0