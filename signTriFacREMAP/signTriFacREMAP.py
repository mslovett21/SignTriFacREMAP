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
    print("sigUppF")
    print("D_pos,F_j")
    print(D_pos.shape)
    print(F_j.shape)
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




def signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, alpha, beta, bal_vec, max_iter):
    
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
        F_vec.append(np.random.rand(layers_size[i],layers_size[i]))

    #initialize T[i] is a diagonal matrix of A[i] 
    T_vec     = []
    for i in range(num_layers):
        T_vec.append(sparse.diags(np.sum(A_vec[i],1),0))
    
    #initialize P_pos_vec that stores low-rank matrices that describe interactions between layers
    P_vec = []
    for i in range(num_pairs_one_inter):
        P_vec.append(np.random.rand(layers_size[x[i]],layers_size[y[i]]))

    #initialize P_pos_vec that stores low-rank matrices that describe interactions between layers
    P_pos_vec = []
    for j in range(num_pairs_sign_inter):
        P_pos_vec.append(np.random.rand(layers_size[sign_x[j]],layers_size[sign_y[j]]))
    
    #initialize P_neg_vec that stores low-rank matrices that describe interactions between layers
    P_neg_vec = []
    for k in range(num_pairs_sign_inter):
        P_neg_vec.append(np.random.rand(layers_size[sign_x[k]],layers_size[sign_y[k]]))
        
    curr_iter        = 0

    print("Finished initialization of the low-rank matrices...")

    while( curr_iter < max_iter):
        
        print(curr_iter)
        one_sided_offset = 0
        signed_offset    = 0
        one_sided_inter  = []
        signed_inter     = []
        print("Current iteration %d:" % (curr_iter))

        for layer_num in range(num_layers):
            #update low-rank layer representation
            upper_sum       = 0
            lower_sum       = 0
            print("Currently investigating relationships with a layer %d:" % (layer_num))
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
   
    
    return (F_vec,P_vec, P_pos_vec, P_neg_vec)


if __name__== '__main__':
    
    print('Number of arguments:', len(sys.argv), 'arguments.')
    if(len(sys.argv) < 8):
        print("Expected arguments: g_matrix.txt d_matrix.txt a_matrix.txt w_matrix.txt max_iteration alpha beta [optional] b_matrix.txt")
    else:
        G_matrix_fh = sys.argv[1]
        G_matrix    = np.loadtxt(open(G_matrix_fh, "rb"), delimiter=",")
        D_matrix_fh = sys.argv[2]
        D_matrix    = np.loadtxt(open(D_matrix_fh, "rb"),dtype='str', delimiter=",")
        A_matrix_fh = sys.argv[3]
        A_matrix    = np.loadtxt(open(A_matrix_fh, "rb"),dtype='str', delimiter=",")       
        W_matrix_fh = sys.argv[4]
        W_matrix    = np.loadtxt(open(W_matrix_fh, "rb"), delimiter=",")
        max_iter    = int(sys.argv[5])
        alpha       = float(sys.argv[6])
        beta        = float(sys.argv[7])

        # OPTIONAL PARAMETER
        b_param_flag = False
        B_matrix = []

        if(len(sys.argv) == 9):
            b_param_flag = True
            B_matrix_fh  = sys.argv[8] 
            B_matrix     = np.loadtxt(open(B_matrix_fh, "rb"), delimiter=",")

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
            D = np.loadtxt(open(abs_file_path,"rb"), delimiter=",")
            print(str(D_matrix[x[i]][y[i]]))
            print(D.shape)
            D_vec.append(D)
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
            print(str(D_matrix[sign_x[j]][sign_y[j]]))
            print(D_pos.shape)
            print(D_neg.shape)
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
            print(A_matrix[k][k])
            A = np.array(list(csv.reader(open(abs_file_path), quoting=csv.QUOTE_NONNUMERIC)))
            A_vec.append(A)
            print(A.shape)

        [F_vec,P_vec, P_pos_vec, P_neg_vec] = signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, alpha, beta, b_vec, max_iter)

        # Save each matrix in separate file
        for i in range(len(F_vec)):
            np.savetxt('F_results/REMAP_pred_F'+str(i)+'.csv',F_vec[i],delimiter=',')

        num_layers       = len(A_vec)
        one_sided_offset = 0
        signed_offset    = 0
        for layer_num in range(num_layers):
            one_sided_inter = np.where(G_matrix[layer_num] == 1)[0]
            signed_inter    = np.where(G_matrix[layer_num] == 2)[0]
            if(len(one_sided_inter) != 0 | len(signed_inter)!= 0):
                for i in range(len(one_sided_inter)):
                    curr = i + one_sided_offset
                    np.savetxt('P_results/REMAP_P_relation'+str(layer_num)+str(one_sided_inter[i]) + '.csv',P_vec[curr],delimiter=',')
                for j in range(len(signed_inter)):
                    curr      = j + signed_offset
                    np.savetxt('P_results/REMAP_P_pos_relation.' + str(layer_num) + str(signed_inter[j]) + '.csv',P_pos_vec[curr],delimiter=',')
                    np.savetxt('P_results/REMAP_P_neg_relation.' + str(layer_num) + str(signed_inter[j]) + '.csv',P_neg_vec[curr],delimiter=',')            




