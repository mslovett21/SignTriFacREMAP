#  G_matrix  - is a upper traingle square numpy matrix that describes relations between layers
#            if: G_matrix[u][v] == 0,  there is no observed/known relation
#                G_matrix[u][v] == 1,  there is a one-sided relation between the layers u and v
#                G_matrix[u][v] == 2,  there is a signed relation, "+" and "-" between the layers u and v [FORMAT: numpy matrix]
#  A_vec       - contains n square matrices, where n is number of layers in the network, each square matrix describes
#                within-layer connectivity (FORMAT: array of arrays)
#  D_vec       - contains matrices that describe one-sided relations between elements of two different layers,
#                if layers u (with n elements) and v (with m elements) have a one-sided relation, then D_vec contains
#                a n by m matrix that describes the relation, if the relation is signed, D_pos_vec contains matrix
#                that describes "+" relation while D_neg_vec contains matrix that describes "-"  relation.
#  D_pos_vec   - contains matrices that describe "-" relation between two layers in the network
#  D_neg_vec   - contains matrices that describe "-" relation between two layers in the network
#  weight_vec  - contains global weight "w" for each of the inter-layer relations
#  alpha       - regularization parameter
#  beta        - regularization parameter
#  bal_vec     - contains balancing parmeter "b" for each of the signed inter-layer relations
#  max_iter    - maximum number of iterations


import numpy as np
from scipy import sparse
from math import pow
import sys
import os
import csv
import scipy as sp
from sklearn import metrics
import time
from scipy.sparse import rand

def D_estimation(D_est, D):

    #observed relations in D
    observed_ind  = list(np.transpose(D.nonzero()))
    #predicted values in D_est
    pred_data     = []
    #collect common indexes
    for j in observed_ind:
        pred_data.append(D_est[j[0],j[1]])
    row,col       = D.nonzero()

    return row,col, pred_data


def oneSidedLowF(F_j, P,F_i,FjtP, weight,D):
    
    w_sq          = pow(weight,2)
    tFj           = F_j.transpose()
    FiP           = F_i.dot(P)
    FiPtFj        = FiP.dot(tFj)
    row,col,data  = D_estimation(FiPtFj,D)
    #needs to be updated to zero out all unobserved entries
    D_tilde       = sparse.csc_matrix((data,(row,col)),shape = D.shape)
    B             = (1-w_sq)* (D_tilde.dot(FjtP)) + w_sq*(FiPtFj.dot(FjtP))
    B_sparse      = sparse.csr_matrix(B)
    
    return B_sparse

def sigUppF(D_pos,D_neg,FjtP_pos,FjtP_neg,bal):

    DposFtP       = D_pos.dot(FjtP_pos)
    DnegFtP       = D_neg.dot(FjtP_neg)
    sig_sum       = (bal*DposFtP) + (1-bal)*DnegFtP
    sig_sum_sparse= sparse.csr_matrix(sig_sum)
    end           = time.time()

    return sig_sum_sparse

def signLowF(D_pos,D_neg,F_j,P_pos,P_neg,FjtP_pos, FjtP_neg, bal,F_i,weight):

    w_sq          = pow(weight,2)
    tFj           = F_j.transpose()
    P_postFj      = P_pos.dot(tFj)
    P_negtFj      = P_neg.dot(tFj)
    FiP_postFj    = F_i.dot(P_postFj)
    row,col,datap = D_estimation(FiP_postFj,D_pos)
    D_pos_tilde   = sparse.csr_matrix((datap,(row,col)),shape = D_pos.shape)
    FiP_negtFj    = F_i.dot(P_negtFj)
    row,col,datan = D_estimation(FiP_negtFj,D_neg)
    D_neg_tilde   = sparse.csr_matrix((datan,(row,col)),shape = D_neg.shape)
    DFjtP_pos     = D_pos_tilde.dot(FjtP_pos)
    DFjtP_neg     = D_neg_tilde.dot(FjtP_neg)

    positive      = bal*((1-w_sq)*(DFjtP_pos) + w_sq*(FiP_postFj.dot(FjtP_pos)))
    negative      = (1-bal)*((1-w_sq)*(DFjtP_neg) + w_sq*(FiP_negtFj.dot(FjtP_neg)))
    lower_sum     = positive + negative
    lower_sum_sp  = sparse.csr_matrix(lower_sum)
    
    return lower_sum_sp

def updateP(F_i, D,F_j, P, weight):

    tFi           = F_i.transpose() 
    A             = tFi.dot(D).dot(F_j)
    w_sq          = pow(weight,2)
    FiPtFj        = F_i.dot(P).dot(F_j.transpose())
    row,col,data  = D_estimation(FiPtFj,D)
    D_tilde       = sparse.csr_matrix((data, (row,col)), shape = D.shape)
    midd          = (1- w_sq)*D_tilde + (w_sq*FiPtFj)
    B             = tFi.dot(midd).dot(F_j)
    B             = B.power(-1)
    A_dividedby_B = A.multiply(B)
    newP          = P.multiply(A_dividedby_B.sqrt())
    newP_sp       = sparse.csr_matrix(newP)
    return newP_sp


def get_predicted_values(masked_indexes, predicted_matrix):
    
    predicted_values = []
    predicted_matrix = predicted_matrix.toarray()
    for i in range(len(masked_indexes)):
        predicted_values.append(predicted_matrix[masked_indexes[i][0]][masked_indexes[i][1]])
    
    return np.array(predicted_values)


def signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, bal_vec, max_iter,alpha,beta):
    start = time.time()
    #number of layers in the network
    num_layers  = len(A_vec)    
    #initialize a vector that stores number of elems in each of the layers
    layers_size = [0] * num_layers
    
    for i in range(num_layers):
        layers_size[i] = A_vec[i].shape[0]
             
    # number of interating pairs of layers
    num_pairs_one_inter  = len(D_vec)
    # number of signed interacting pairs
    num_pairs_sign_inter = len(D_neg_vec) 
    #get indexes of all interacting pairs of layers from G_matrix
    [x,y]                = (np.argwhere(G_matrix == 1)).T
    #get indexes of signed pair interactions
    [sign_x,sign_y]      = (np.argwhere(G_matrix == 2)).T

    #initialize low-rank representation for each of the layers
    F_vec     = []
    for i in range(num_layers):
        F_vec.append(rand(layers_size[i],layers_size[i], density=1.0, format='csr'))

    #initialize T[i] is a diagonal matrix of A[i] 
    T_vec     = []
    for i in range(num_layers):
        A_diagonals = np.array(A_vec[i].sum(1))
        diagonals = [item[0] for item in A_diagonals]
        size_diag= len(diagonals)
        T_vec.append(sparse.diags(diagonals,0,shape=(size_diag,size_diag),format="csr"))
    
    #initialize P_pos_vec that stores low-rank matrices that describe interactions between layers
    P_vec = []
    for i in range(num_pairs_one_inter):
        P_vec.append(rand(layers_size[x[i]],layers_size[y[i]], density = 1.0, format = 'csr'))

    #initialize P_pos_vec that stores low-rank matrices that describe interactions between layers
    P_pos_vec = []
    for j in range(num_pairs_sign_inter):
        P_pos_vec.append(rand(layers_size[sign_x[j]],layers_size[sign_y[j]], density=1.0, format='csr'))
    
    #initialize P_neg_vec that stores low-rank matrices that describe interactions between layers
    P_neg_vec = []
    for k in range(num_pairs_sign_inter):
        P_neg_vec.append(rand(layers_size[sign_x[k]],layers_size[sign_y[k]], density = 1.0, format='csr'))   
    end = time.time()
    curr_iter           = 0
    G_size              = G_matrix.shape[0]
    ind                 = np.tril_indices(G_size,-1)
    G_matrix[ind]       = G_matrix.T[ind]


    while( curr_iter < max_iter):
        start  = time.time()
        
        one_sided_offset      = 0
        signed_offset         = 0
        
        l_one_sided_offset    = 0
        l_signed_offset       = 0

        glo_one_sided_offset  = 0
        glo_signed_offset     = 0

        glo_lone_sided_offset = 0
        glo_lsigned_offset    = 0


        one_sided_inter       = []
        signed_inter          = []

        for layer_num in range(num_layers):

            #initialize matrix of shape of a layer that is currently being updated
            upper_sum       = rand(F_vec[layer_num].shape[0], F_vec[layer_num].shape[1], density=0.1, format = "csr")
            lower_sum       = rand(F_vec[layer_num].shape[0], F_vec[layer_num].shape[1], density=0.1, format = "csr")

            one_sided_inter = np.where(G_matrix[layer_num] == 1)[0]
            signed_inter    = np.where(G_matrix[layer_num] == 2)[0]

            for i in range(len(one_sided_inter)):                
                curr      = 0
                if (layer_num > one_sided_inter[i]):
                    curr                   =  glo_lone_sided_offset
                    glo_lone_sided_offset  =  glo_lone_sided_offset + 1
                    l_one_sided_offset     =  l_one_sided_offset + 1
                else:
                    curr                   = glo_one_sided_offset
                    glo_one_sided_offset   = glo_one_sided_offset + 1
                    one_sided_offset       = one_sided_offset + 1
                P         = P_vec[curr]
                D         = D_vec[curr]
                if (layer_num > one_sided_inter[i]):
                    P     = P.transpose()
                    D     = D.transpose()
                tP        = P.transpose()
                FjtP      = F_vec[one_sided_inter[i]].dot(tP)
                upper_sum = upper_sum + D.dot(FjtP)
                lower_sum = lower_sum + oneSidedLowF( F_vec[one_sided_inter[i]], P,F_vec[layer_num],FjtP, weight_vec[curr], D)
            
            for j in range(len(signed_inter)):
                curr      = 0

                if(layer_num > signed_inter[j]):
                    curr               = glo_lsigned_offset
                    glo_lsigned_offset = glo_lsigned_offset + 1
                    l_signed_offset    = l_signed_offset + 1
                else:
                    curr              = glo_signed_offset
                    glo_signed_offset = glo_signed_offset + 1
                    signed_offset     = signed_offset + 1
                P_pos     = P_pos_vec[curr]
                P_neg     = P_neg_vec[curr]
                D_pos     = D_pos_vec[curr]
                D_neg     = D_neg_vec[curr]
                if(layer_num < signed_inter[j]):
                    tP_pos    = P_pos.transpose()
                    tP_neg    = P_neg.transpose()
                else:
                    tP_pos    = P_pos
                    tP_neg    = P_neg
                    D_pos     = D_pos.transpose()
                    D_neg     = D_neg.transpose()
                    P_pos     = P_pos.transpose()
                    P_neg     = P_neg.transpose()
                FjtP_pos  = F_vec[signed_inter[j]].dot(tP_pos)
                FjtP_neg  = F_vec[signed_inter[j]].dot(tP_neg)

                upper_sum = sparse.csr_matrix(upper_sum) + sigUppF(D_pos,D_neg,FjtP_pos,FjtP_neg, bal_vec[curr])
                lower_sum = sparse.csr_matrix(lower_sum) + signLowF(D_pos,D_neg,F_vec[signed_inter[j]],P_pos,P_neg,FjtP_pos,FjtP_neg, bal_vec[curr],F_vec[layer_num],weight_sign_vec[curr])
            A = upper_sum + (alpha * A_vec[layer_num].dot(F_vec[layer_num]))
            B = lower_sum + (alpha * T_vec[layer_num].dot(F_vec[layer_num]) + (beta * F_vec[layer_num]))
            B = B.power(-1)
            A_dividedby_B = A.multiply(B)
            F_vec[layer_num] = F_vec[layer_num].multiply(A_dividedby_B.sqrt())

            #update low-rank inter-layer relation matrices that involve that layer
            start_one_sided     = glo_one_sided_offset - one_sided_offset
            end_one_sided       = glo_one_sided_offset

            for k in range(len(one_sided_inter)):
                if(layer_num > one_sided_inter[k]):
                    pass
                else:
                    if(start_one_sided < end_one_sided):
                        curr            = start_one_sided
                        P_vec[curr]     = updateP(F_vec[layer_num], D_vec[curr],F_vec[one_sided_inter[k]], P_vec[curr], weight_vec[curr])
                        start_one_sided = start_one_sided + 1
            start_signed        = glo_signed_offset - signed_offset
            end_signed          = glo_signed_offset 
            for l in range(len(signed_inter)):
                if (layer_num > signed_inter[l]):
                    pass
                else:
                    if(start_signed < end_signed):
                        curr            = start_signed
                        P_pos_vec[curr] = updateP(F_vec[layer_num], D_pos_vec[curr],F_vec[signed_inter[l]], P_pos_vec[curr], weight_sign_vec[curr])
                        P_neg_vec[curr] = updateP(F_vec[layer_num], D_neg_vec[curr],F_vec[signed_inter[l]], P_neg_vec[curr], weight_sign_vec[curr])
                        start_signed    = start_signed + 1
            
            one_sided_offset   = 0
            signed_offset      = 0
            l_one_sided_offset = 0
            l_signed_offset    = 0
  

        curr_iter        = curr_iter + 1
        end = time.time()
        print("Time to complete the interations:")
        print(end - start)
    

    return (F_vec,P_vec, P_pos_vec, P_neg_vec)


        
def signTriFacREMAP_CV(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter):


    print("Start the 10 fold CV for:weight_vec, weight_sign_vec, alpha, beta, bal_vec")

    start_fold    = time.time()
    end_fold      = time.time()
    alpha_vec     = []
    beta_vec      = []
    P_vec         = []
    F_vec         = []
    P_pos_vec     = []
    P_neg_vec     = []
    D_vec         = []

    alphas_range  = np.arange(0.1,1.1,0.2)
    betas_range   = np.arange(0.1,1.1,0.2)  
    weights_range = np.arange(0.1,1.1,0.2)
    G_original = np.copy(G_matrix)
    file_time = open("size_100_iter50_time.txt", "a+")

    for i in range(len(alphas_range)):
        alpha = alphas_range[i]
        for j in range(len(betas_range)):
            beta = betas_range[j]
            for k in range(len(weights_range)):
                weight_vec[0] = weights_range[k]
                for l in range(len(weights_range)):
                    weight_sign_vec[0] = weights_range[l]
                    for m in range(len(weights_range)):
                        weight_sign_vec[1] = weights_range[m]
                        print("Current alpha: %.2f, beta: %.2f, w[0]: %.2f, sign_w[0]: %.2f, sign_w[1]: %.2f" % (alpha,beta, weight_vec[0], weight_sign_vec[0], weight_sign_vec[1]))
                        # here we are opening 10 different files containing different matrices together with their associated indexes files.
                        # indexes should be stored in the list of lists
                        for k in range(1,11):
                            start_fold= time.time()
                            D_vec.append(sparse.load_npz("data/ten_fold/matrix_fold_" + str(k) + ".npz"))
                            print("FOLD: %d" % ( k))
                            G_matrix =np.copy(G_original)
                            F_vec,P_vec, P_pos_vec, P_neg_vec = signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter,alpha,beta)
                            predicted_matrix                  = np.dot(np.dot(F_vec[0],P_vec[0]), F_vec[2].transpose())
                            masked_indexes = []
                            #open the file with the indexes
                            file = open('data/ten_fold/index_fold_' + str(k) + '.csv', 'r')
                            for line in file:
                                values    = line.strip('{}\n\r ').split(',')
                                values[0] = int(values[0])
                                values[1] = int(values[1])
                                masked_indexes.append(values)
                            file.close()
                            predicted_values                  = get_predicted_values(masked_indexes, predicted_matrix)
                            #open file to store the predicted values
                            pred_val_file = open('sparse_signTriFac/pred_val_' + str(k) + "alpha"+ str(alpha)+ "beta"+ str(beta)+"w"+str(weight_vec[0])+"w"+str(weight_sign_vec[0]) +"w"+str(weight_sign_vec[1])+ '.csv','a')
                            for v in range(len(predicted_values)):
                                pred_val_file.write(str(predicted_values[v]))
                                pred_val_file.write("\n")
                            pred_val_file.close()
                            D_vec = []
                            end_fold = time.time()
                            file_time.write("Time spend on fold "+ str(k) + ": " + str(end_fold-start_fold))
                            file_time.write("\n")
                            file_time.flush()




    return F_vec,P_vec, P_pos_vec, P_neg_vec 
           


      


if __name__== '__main__':

    print("This is 10-fold Cross-Validation for: alpha, beta and w_matrix")
    start = time.time()
    print('Number of provided argumenents %d' %  (len(sys.argv)-1))
    if(len(sys.argv) < 5):
        print("Expected arguments: g_matrix.txt d_matrix.txt a_matrix.txt max_iteration ")
    else:
        G_matrix_fh = sys.argv[1]
        G_matrix    = np.loadtxt(open(G_matrix_fh, "rb"), delimiter=",")
        D_matrix_fh = sys.argv[2]
        D_matrix    = np.loadtxt(open(D_matrix_fh,"rb"),dtype="str", delimiter=",")
        A_matrix_fh = sys.argv[3]
        A_matrix    = np.loadtxt(open(A_matrix_fh, "rb"),dtype='str', delimiter=",")       
        max_iter    = int(sys.argv[4])
        b_param_flag = False


        #number of layers in the problem
        num_layers = G_matrix.shape[0]   
        #get indexes of all one-sided relations between layers from G_matrix
        [x,y]                = (np.argwhere(G_matrix == 1)).T
        [sign_x,sign_y]      = (np.argwhere(G_matrix == 2)).T  
        weight_vec           = []
        weight_sign_vec      = []
        b_vec                = []      
        #get the absolute path to the script
        script_dir = os.path.abspath(__file__) #<-- absolute dir the script is in
        script_dir = os.path.split(script_dir)[0] #i.e. /path/to/dir/
    

        #Load all D matrices describing one-sided relations into D_vec
        D_vec=[]
        for i in range(len(x)):
            abs_file_path = script_dir + str(D_matrix[x[i]][y[i]])
            D = sparse.load_npz(abs_file_path)
            D_vec.append(D)
            weight_vec.append(1.0)

        #Load all D matrices describing signed relations into D_pos_vec and D_neg_vec
        D_pos_vec = []
        D_neg_vec = []
        for j in range(len(sign_x)):
            abs_file_path = script_dir + str(D_matrix[sign_x[j]][sign_y[j]])
            D     = sparse.load_npz(abs_file_path)
            D     = D.todense()
            D_pos = 0.5 * (np.abs(D) + D)
            D_neg = 0.5 * (np.abs(D) - D)
            D_pos_vec.append(sparse.csr_matrix(D_pos))
            D_neg_vec.append(sparse.csr_matrix(D_neg))
            weight_sign_vec.append(1.0)

            if(b_param_flag):
                b_vec.append(B_matrix[sign_x[j]][sign_y[j]])
            else:
                b_vec.append(0.5)

        #Load all A matrices 
        A_vec = []       
        for k in range(num_layers):
            abs_file_path = script_dir + str(A_matrix[k][k])
            A = sparse.load_npz(abs_file_path)
            A_vec.append(A)
        [F_vec,P_vec, P_pos_vec, P_neg_vec] = signTriFacREMAP_CV(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter)
        end = time.time()
        print(end - start)


