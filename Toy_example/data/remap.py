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
from sklearn import metrics
import time
from scipy.sparse import rand



def getCurrD(F_i,P,F_j):
    
    FiP   = np.dot(F_i,P)
    currD = np.dot(FiP, F_j.transpose())
    
    return currD

def oneSidedUppF(D, F_j, P):

    DF   = np.dot(D,F_j)
    DFtP = np.dot(DF,P.transpose())

    return DFtP

def oneSidedLowF(F_j, P,F_i, weight):
    
    w_sq      = pow(weight,2)
    D_est     = getCurrD(F_i,P,F_j)   
    FjtP      = np.dot(F_j,P.transpose())
    FiP       = np.dot(F_i,P)
    tFj       = F_j.transpose()
    B = (1-w_sq)*np.dot(D_est,FjtP) + w_sq*(np.dot(np.dot(FiP,tFj),FjtP))

    return B

def sigUppF(D_pos,D_neg,F_j,P_neg,P_pos,bal):
    DposF   = np.dot(D_pos,F_j)
    DposFtP = np.dot(DposF, P_pos.transpose())
    DnegF   = np.dot(D_neg,F_j)
    DnegFtP = np.dot(DnegF, P_neg.transpose())
    sig_sum =  (bal*DposFtP) + (1-bal)*DnegFtP  
    return sig_sum

def signLowF(D_pos,D_neg,F_j,P_pos,P_neg, bal,F_i,weight):
   
    w_sq          = pow(weight,2)
    D_pos_est     = getCurrD(F_i,P_pos,F_j)
    D_neg_est     = getCurrD(F_i,P_neg,F_j)
    tFj           = F_j.transpose()
    FjtP_pos      = np.dot(F_j,P_pos.transpose())
    FjtP_neg      = np.dot(F_j,P_neg.transpose())
    P_postFj      = np.dot(P_pos,tFj)
    P_negtFj      = np.dot(P_neg,tFj)
    
    DFjtP_pos     = np.dot(D_pos_est,FjtP_pos)
    DFjtP_neg     = np.dot(D_neg_est,FjtP_neg) 

    positive      = bal*((1-w_sq)*(DFjtP_pos) + w_sq*(np.dot(F_i,np.dot(P_postFj,FjtP_pos))))
    negative      = (1-bal)*((1-w_sq)*(DFjtP_neg) + w_sq*(np.dot(F_i,np.dot(P_negtFj,FjtP_neg))))
    lower_sum     = positive + negative

    return lower_sum

def updateP(F_i, D,F_j, P, weight):

    tFi    = F_i.transpose() 
    A      = np.dot(np.dot(tFi,D), F_j)
    D_est  = getCurrD(F_i,P,F_j)
    w_sq   = pow(weight,2)
    FiPtFj = np.dot(np.dot(F_i,P),F_j.transpose())
    midd   = (1- w_sq)*D_est + (w_sq*FiPtFj)
    B      = np.dot(np.dot(tFi,midd),F_j)
    newP   = np.multiply(P,np.sqrt(np.divide(A,B)))
    
    return B


def get_predicted_values(masked_indexes, predicted_matrix):
    predicted_values = []
    for i in range(len(masked_indexes)):
        predicted_values.append(predicted_matrix.item(masked_indexes[i][0], masked_indexes[i][1]))
    return np.array(predicted_values)





def signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, bal_vec, max_iter,alpha,beta):
    
    #check if the data makes sense- meets the requirements
    #FILL IN LATER
    
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
        T_vec.append(sparse.diags(np.sum(A_vec[i],1),0))
    
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

    curr_iter        = 0
    start = time.time()
    while( curr_iter < max_iter):
        print("Current iteration %d" % (curr_iter))
        
        one_sided_offset = 0
        signed_offset    = 0
        one_sided_inter  = []
        signed_inter     = []

        for layer_num in range(num_layers):
            #update low-rank layer representation
            upper_sum       = 0
            lower_sum       = 0

            one_sided_inter = np.where(G_matrix[layer_num] == 1)[0]
            signed_inter    = np.where(G_matrix[layer_num] == 2)[0]
            for i in range(len(one_sided_inter)):
                curr      = i + one_sided_offset
                upper_sum = oneSidedUppF(D_vec[curr], F_vec[one_sided_inter[i]], P_vec[curr])
                lower_sum = lower_sum + oneSidedLowF( F_vec[one_sided_inter[i]], P_vec[curr],F_vec[layer_num], weight_vec[curr])
            for j in range(len(signed_inter)):
                curr = j + signed_offset
                upper_sum = upper_sum + sigUppF(D_pos_vec[curr],D_neg_vec[curr],F_vec[signed_inter[j]],P_pos_vec[curr],P_neg_vec[curr], bal_vec[curr])
                lower_sum = lower_sum + signLowF(D_pos_vec[curr],D_neg_vec[curr],F_vec[signed_inter[j]],P_pos_vec[curr],P_neg_vec[curr], bal_vec[curr],F_vec[layer_num],weight_sign_vec[curr])
                
            A = upper_sum + (alpha * np.dot(A_vec[layer_num],F_vec[layer_num]))
            B = lower_sum + (alpha * np.dot(T_vec[layer_num].todense(),F_vec[layer_num])) + (beta * F_vec[layer_num]) 
            F_vec[layer_num] = np.multiply(F_vec[layer_num],np.sqrt(np.divide(A,B)))
             
                #update low-rank inter-layer relation matrices that involve that layer
            for k in range(len(one_sided_inter)):
                curr            = k + one_sided_offset
                P_vec[curr]     = updateP(F_vec[layer_num], D_vec[curr],F_vec[one_sided_inter[k]], P_vec[curr], weight_vec[curr])
            for l in range(len(signed_inter)):
                curr            = l + signed_offset
                P_pos_vec[curr] = updateP(F_vec[layer_num], D_pos_vec[curr],F_vec[signed_inter[l]], P_pos_vec[curr], weight_sign_vec[curr])
                P_neg_vec[curr] = updateP(F_vec[layer_num], D_neg_vec[curr],F_vec[signed_inter[l]], P_neg_vec[curr], weight_sign_vec[curr])
                
            one_sided_offset = one_sided_offset + len(one_sided_inter)
            signed_offset    = signed_offset    + len(signed_inter)

        curr_iter        = curr_iter + 1
    end = time.time()
    print("Time to complete the interations:")
    print(end - start)
   
    
    return (F_vec,P_vec, P_pos_vec, P_neg_vec)


        
def signTriFacREMAP_CV(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter):


    print("Start the 10 fold CV for:weight_vec, weight_sign_vec, alpha, beta, bal_vec")

    filep = open("./results/results_d_vec0.txt", "w+")

    check = open("check.txt", "w+")

    alpha_vec     = []
    beta_vec      = []
    P_vec         = []
    F_vec         = []
    P_pos_vec     = []
    P_neg_vec     = []

    alphas_range  = np.arange(0.1,1.1,0.1)
    betas_range   = np.arange(0.1,1.1,0.1)  
    weights_range = np.arange(0.1,1.1,0.1)
    
    original_matrix = D_vec[0].copy()

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
                            D_vec[0]                          = sparse.load_npz("ten_fol/matrix_fold_" + str(k) + ".npz")
                            F_vec,P_vec, P_pos_vec, P_neg_vec = signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter,alpha,beta)
                            predicted_matrix                  = np.dot(np.dot(F_vec[0],P_vec[0]), F_vec[2].transpose())
                            masked_indexes = []
                            #open the file with the indexes
                            file = open('ten_fold/index_fold_' + str(k) + '.csv', 'r')
                            for line in file:
                                values    = line.strip('{}\n\r ').split(',')
                                values[0] = int(values[0])
                                values[1] = int(values[1])
                                masked_indexes.append(values)
                            predicted_values                  = get_predicted_values(masked_indexes, predicted_matrix)
                            #open file to store the predicted values
                            pred_val_file = open('ten_fold/pred_val_' + str(k) + '.csv','a')
                            for v in range(len(predicted_values)):
                                pred_val_file.write(str(predicted_values[v]))
                                pred_val_file.write("\n")
                            mask_file.close()
                            


                        filep.write("alpha: %.2f, beta: %.2f,w[0]: %.2f, sign_w[0]: %.2f, sign_w[1]: %.2f, precision: %.2f, auc: %.2f, recall: %.2f, f1: %.2f \n" % (alpha,beta, weight_vec[0], weight_sign_vec[0], weight_sign_vec[1],precision_vec[-1],auc_vec[-1],recall_vec[-1],f1_vec[-1]))
                        filep.flush()
                        os.fsync(filep)



    return F_vec,P_vec, P_pos_vec, P_neg_vec 
           


      


if __name__== '__main__':

    print("This is 10-fold Cross-Validation for: w_weight, alpha, beta and w_matrix")
    
    print('Number of arguments:', len(sys.argv), 'arguments.')
    if(len(sys.argv) < 5):
        print("Expected arguments: g_matrix.txt d_matrix.txt a_matrix.txt max_iteration w_matrix ")
    else:
        G_matrix_fh = sys.argv[1]
        G_matrix    = np.loadtxt(open(G_matrix_fh, "rb"), delimiter=",")
        D_matrix_fh = sys.argv[2]
        D_matrix    = np.loadtxt(open(D_matrix_fh,"rb"), delimiter=",")
        A_matrix_fh = sys.argv[3]
        A_matrix    = np.loadtxt(open(A_matrix_fh, "rb"),dtype='str', delimiter=",")       
        max_iter    = int(sys.argv[4])
        W_matrix_fh = sys.argv[5]
        W_matrix    = np.loadtxt(open(W_matrix_fh, "rb"), delimiter=",")
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
            curr_weight = W_matrix[x[i]][y[i]]
            weight_vec.append(curr_weight)

        #Load all D matrices describing signed relations into D_pos_vec and D_neg_vec
        D_pos_vec = []
        D_neg_vec = []
        for j in range(len(sign_x)):
            abs_file_path = script_dir + str(D_matrix[sign_x[j]][sign_y[j]])
            D     = np.array(list(csv.reader(open(abs_file_path), quoting=csv.QUOTE_NONNUMERIC)))
            D_pos = 0.5 * (np.abs(D) + D)
            D_neg = 0.5 * (np.abs(D) - D)
            D_pos_vec.append(D_pos)
            D_neg_vec.append(D_neg)
            curr_weight = W_matrix[sign_x[j]][sign_y[j]]
            weight_sign_vec.append(curr_weight)

            if(b_param_flag):
                b_vec.append(B_matrix[sign_x[j]][sign_y[j]])
            else:
                b_vec.append(0.5)

        #Load all A matrices 
        A_vec = []       
        for k in range(num_layers):
            abs_file_path = script_dir + str(A_matrix[k][k])
            A = np.array(list(csv.reader(open(abs_file_path), quoting=csv.QUOTE_NONNUMERIC)))
            A_vec.append(A)


        [F_vec,P_vec, P_pos_vec, P_neg_vec] = signTriFacREMAP_CV(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter)


