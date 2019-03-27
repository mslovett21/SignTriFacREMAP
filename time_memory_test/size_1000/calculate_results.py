import pandas as pd 
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score






if __name__=='__main__':
    
    #File to store the results
    file_time = open("size_100_iter50_time_results.txt", "a+") 
    
    #This is not automated yet for all the values of the parameters because not all of the results are avaiable
    #possible values of the parameters
    possible_values =[0.1,0.30000000000000004,5000000000000001,0.7000000000000001,0.9000000000000001]
    
    #current setting of the parameters -setting them should be automoated by a foor loop
    alpha_val = str(0.1)
    beta_val  = str(0.1)
    w0        = str(0.1)
    w1        = str(0.1)
    w2        = str(0.7000000000000001)

    # PREDICTED VALUES FOR THE FOLD
    file_name_part2 = 'alpha'+alpha_val+'beta'+beta_val+'w'+w0+'w'+w1+'w'+w2+'.csv'
    file_name_part2 = 'alpha0.1beta0.1w0.1w0.1w0.7.csv' 
    auc_scores_for_folds = []
    map_scores_for_folds = []

    #Loop through all the folds
    for i in range(1,11):
        fold_num = i
        # True values for the fold
        path_to_true_values      =   'data/ten_fold/masked_val_fold_' + str(fold_num) +'.csv'
        masked_val               =   pd.read_csv(path_to_true_values, sep = '\n')
        masked_values_array      =   masked_val.values
        # PREDICTED VALUES FOR THE FOLD
        file_name                =   'pred_val_'+str(fold_num)+ file_name_part2
        path_to_predicted_values =   'sparse_SignTriFac/' + file_name
        predicted_values         =   pd.read_csv(path_to_predicted_values, sep ='\n')
        predicted_values_array   =   predicted_values.values
        #area under ROC curve
        fpr, tpr, thresholds     =   metrics.roc_curve(masked_values_array, predicted_values_array)
        fold_auc_score          =   metrics.auc(fpr, tpr)
        # MAP soore mean average precision object detection
        fold_map_score           =   average_precision_score(masked_values_array, predicted_values_array) 
        auc_scores_for_folds.append(fold_auc_score)
        map_scores_for_folds.append(fold_map_score)
        print(fold_auc_score)
   
   #Mean performance for all 10 folds
    map_ave = np.mean(map_scores_for_folds)
    auc_ave = np.mean(map_scores_for_folds)
    file_time.write(file_name + ": " +"ave_auc: "+ str(auc_ave) + "map_ave: " + str(map_ave))
    file_time.write("\n")
    file_time.flush()




