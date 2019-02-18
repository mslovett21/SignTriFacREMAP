# SignTriFac

## Neighborhood-Regularized Tri-Factorization One-Class Collaborative Filtering Algorithm for Signed Networks.

### To run:

```
python signTriFacREMAP.py conf_files/g_matrix.txt conf_files/d_matrices.txt conf_files/a_matrices.txt conf_files/w_matrix 50 0.6 0.5

```

### To see example of data and check required data formats check Toy_examle folder.

## The scripts in this repo:

- **data_mat.py** - this scripts reads in original data files in matlab format and transforms them into appropriate txt files

- **signTriFacREMAP.py** - not optimaized for memory or parralel processing version of the code, suitable only for small matrices.

- **signTriFacREMAP_CV.py** version of the code above that inludes 10 fold cross validation of the hyperparameters

- **signTriFac_CVsparse.py** version of the above code with the use of sparse matrices and offline 10 fold validation (pre-prepared) 
 
