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



def getCurrD(F_i,P,F_j):
    FiP   = F_i.dot(P)
    currD = FiP.dot(F_j.transpose())
    return sparse.csr_matrix(currD)

def oneSidedUppF(D, F_j, P):
    DF   = D.dot(F_j)
    DFtP = DF.dot(P.transpose())
    print("oneSidedUppF returns shape")
    print(DFtP.shape)
    return sparse.csr_matrix(DFtP)

def oneSidedLowF(F_j, P,F_i, weight):
    w_sq      = pow(weight,2)
    D_est     = getCurrD(F_i,P,F_j)   
    FjtP      = F_j.dot(P.transpose())
    FiP       = F_i.dot(P)
    tFj       = F_j.transpose()
    B = (1-w_sq)* (D_est.dot(FjtP)) + w_sq*(FiP.dot(tFj).dot(FjtP))
    print("one sided LowF returns shape")
    print(B.shape)
    return sparse.csr_matrix(B)

def sigUppF(D_pos,D_neg,F_j,P_neg,P_pos,bal):
    DposF   = D_pos.dot(F_j)
    DposFtP = DposF.dot(P_pos.transpose())
    DnegF   = D_neg.dot(F_j)
    DnegFtP = DnegF.dot(P_neg.transpose())
    sig_sum =  (bal*DposFtP) + (1-bal)*DnegFtP
    print("sigUppF shape")
    print(sig_sum.shape)
    return sparse.csr_matrix(sig_sum)

def signLowF(D_pos,D_neg,F_j,P_pos,P_neg, bal,F_i,weight):
   
    w_sq          = pow(weight,2)
    D_pos_est     = getCurrD(F_i,P_pos,F_j)
    D_neg_est     = getCurrD(F_i,P_neg,F_j)
    tFj           = F_j.transpose()
    FjtP_pos      = F_j.dot(P_pos.transpose())
    FjtP_neg      = F_j.dot(P_neg.transpose())
    P_postFj      = P_pos.dot(tFj)
    P_negtFj      = P_neg.dot(tFj)
    
    DFjtP_pos     = D_pos_est.dot(FjtP_pos)
    DFjtP_neg     = D_neg_est.dot(FjtP_neg) 

    positive      = bal*((1-w_sq)*(DFjtP_pos) + w_sq*(F_i.dot(P_postFj.dot(FjtP_pos))))
    negative      = (1-bal)*((1-w_sq)*(DFjtP_neg) + w_sq*(F_i.dot(P_negtFj.dot(FjtP_neg))))
    lower_sum     = positive + negative
    print("signLowF shape")
    print(lower_sum.shape)
    return sparse.csr_matrix(lower_sum)

def updateP(F_i, D,F_j, P, weight):
    tFi    = F_i.transpose() 
    A      = tFi.dot(D).dot(F_j)
    D_est  = getCurrD(F_i,P,F_j)
    w_sq   = pow(weight,2)
    FiPtFj = F_i.dot(P).dot(F_j.transpose())
    midd   = (1- w_sq)*D_est + (w_sq*FiPtFj)
    B      = tFi.dot(midd).dot(F_j)
    B = B.power(-1)
    A_dividedby_B = A.multiply(B)
    newP   = P.multiply(A_dividedby_B.sqrt())
    return sparse.csr_matrix(newP)


def get_predicted_values(masked_indexes, predicted_matrix):
    print("IN THE GET PREDICTED VALUES")
    predicted_values = []
    predicted_matrix = predicted_matrix.toarray()
    for i in range(len(masked_indexes)):
        predicted_values.append(predicted_matrix[masked_indexes[i][0]][masked_indexes[i][1]])
    return np.array(predicted_values)





def signTriFacREMAP(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, bal_vec, max_iter,alpha,beta):

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

    curr_iter        = 0
    start = time.time()
    while( curr_iter < max_iter):
        print("Current iteration %d" % (curr_iter))
        
        one_sided_offset = 0
        signed_offset    = 0
        one_sided_inter  = []
        signed_inter     = []

        for layer_num in range(num_layers):
            print("Layer")
            print(layer_num)
            #update low-rank layer representation
            upper_sum       = rand(F_vec[layer_num].shape[0], F_vec[layer_num].shape[1], density=0.1, format="csr")
            lower_sum       = rand(F_vec[layer_num].shape[0], F_vec[layer_num].shape[1], density=0.1, format = "csr")
            print("Initialize upper sum and lower sum")
            print(upper_sum.shape)
            print(lower_sum.shape)

            one_sided_inter = np.where(G_matrix[layer_num] == 1)[0]
            signed_inter    = np.where(G_matrix[layer_num] == 2)[0]
            for i in range(len(one_sided_inter)):
                curr      = i + one_sided_offset
                upper_sum = oneSidedUppF(D_vec[curr], F_vec[one_sided_inter[i]], P_vec[curr])
                print("upper sum in the one sided inter")
                print(upper_sum.shape)
                lower_sum = lower_sum + oneSidedLowF( F_vec[one_sided_inter[i]], P_vec[curr],F_vec[layer_num], weight_vec[curr])
                print("lower sum in the one sided inter")
                print(lower_sum.shape)
            for j in range(len(signed_inter)):
                curr = j + signed_offset
                print("upper sum in the signed inter")
                print(upper_sum.shape)
                upper_sum = sparse.csr_matrix(upper_sum) + sigUppF(D_pos_vec[curr],D_neg_vec[curr],F_vec[signed_inter[j]],P_pos_vec[curr],P_neg_vec[curr], bal_vec[curr])
                print("lower_sum in the signed inter")
                print(lower_sum.shape)
                lower_sum = sparse.csr_matrix(lower_sum) + signLowF(D_pos_vec[curr],D_neg_vec[curr],F_vec[signed_inter[j]],P_pos_vec[curr],P_neg_vec[curr], bal_vec[curr],F_vec[layer_num],weight_sign_vec[curr])
            A = upper_sum + (alpha * A_vec[layer_num].dot(F_vec[layer_num]))
            B = lower_sum + (alpha * T_vec[layer_num].dot(F_vec[layer_num]) + (beta * F_vec[layer_num]))
            B = B.power(-1)
            print("A shape")
            print(A.shape)
            print("B shape")
            print(B.shape)
            print("A_ dividedby_B shape")
            A_dividedby_B = A.multiply(B)
            print(A_dividedby_B.shape)
            F_vec[layer_num] = F_vec[layer_num].multiply(A_dividedby_B.sqrt())
            print("F_vec[layer_num")
            print(F_vec[layer_num].shape)
                #update low-rank inter-layer relation matrices that involve that layer
            for k in range(len(one_sided_inter)):
                curr            = k + one_sided_offset
                P_vec[curr]     = updateP(F_vec[layer_num], D_vec[curr],F_vec[one_sided_inter[k]], P_vec[curr], weight_vec[curr])
            for l in range(len(signed_inter)):
                curr            = l + signed_offset
                P_pos_vec[curr] = updateP(F_vec[layer_num], D_pos_vec[curr],F_vec[signed_inter[l]], P_pos_vec[curr], weight_sign_vec[curr])
                P_neg_vec[curr] = updateP(F_vec[layer_num], D_neg_vec[curr],F_vec[signed_inter[l]], P_neg_vec[curr], weight_sign_vec[curr])
            print("Done with the updates")
            one_sided_offset = one_sided_offset + len(one_sided_inter)
            signed_offset    = signed_offset    + len(signed_inter)

        curr_iter        = curr_iter + 1
        print("Should start the next iteration")
        print(curr_iter)
    end = time.time()
    print("Time to complete the interations:")
    print(end - start)
    print("Exiting the function")
   
    
    return (F_vec,P_vec, P_pos_vec, P_neg_vec)


        
def signTriFacREMAP_CV(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter):


    print("Start the 10 fold CV for:weight_vec, weight_sign_vec, alpha, beta, bal_vec")

    filep = open("data/results/results_d_vec0.txt", "w+")

    check = open("data/check.txt", "w+")

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
                            D_vec.append(sparse.load_npz("data/ten_fold/matrix_fold_" + str(k) + ".npz"))
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
                            pred_val_file = open('data/ten_fold/pred_val_' + str(k) + "alpha"+ str(alpha)+ "beta"+ str(beta)+"w"+str(weight_vec[0])+"w"+str(weight_sign_vec[0]) +"w"+str(weight_sign_vec[1])+ '.csv','a')
                            for v in range(len(predicted_values)):
                                pred_val_file.write(str(predicted_values[v]))
                                pred_val_file.write("\n")
                            pred_val_file.close()
                            D_vec[:] = []
                            


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
        D_matrix    = np.loadtxt(open(D_matrix_fh,"rb"),dtype="str", delimiter=",")
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
        print("print len of x")
        print(len(x))
        for i in range(len(x)):
            abs_file_path = script_dir + str(D_matrix[x[i]][y[i]])
            D = np.loadtxt(open(abs_file_path,"rb"), delimiter=",")
            D_vec.append(sparse.csr_matrix(D))
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
            D_pos_vec.append(sparse.csr_matrix(D_pos))
            D_neg_vec.append(sparse.csr_matrix(D_neg))
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
            A_vec.append(sparse.csr_matrix(A))


        [F_vec,P_vec, P_pos_vec, P_neg_vec] = signTriFacREMAP_CV(G_matrix, A_vec, D_vec, D_pos_vec, D_neg_vec, weight_vec, weight_sign_vec, b_vec, max_iter)


