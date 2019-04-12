#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:13:09 2019

@author: sklarjg
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

from scipy import interp
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import xgboost as xgb


#%%


### LOAD MICROBIOME OTU EXPRESSION DATA
data = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/seqtab_md5_6min_rar20k.txt", sep = "\t")
data.index = data["OTU ID"]
data.drop("OTU ID", axis = 1, inplace = True)
data = data.T
data.sort_index(inplace=True)

##STUDY BIOMARKERS:
biomarker_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/biomarkers.txt", sep = "\t")
biomarker_df.index = biomarker_df["UserNumber"]
biomarker_df.sort_index(inplace=True)


### LOAD PATIENT METADATA
metadata = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/180904_Dutch samples manifest_IVC3_mapforR.txt", sep = "\t")
metadata.index = metadata["sampID"]
metadata.sort_index(inplace=True)

###REMOVE METADATA WITH NO SAMPLES
remove_from_meta = np.setdiff1d(metadata.index, data.index)
metadata.drop(remove_from_meta, axis = 0, inplace = True)

###REMOVE SAMPLES WITH NO METADATA
remove_from_data = np.setdiff1d(data.index, metadata.index)
data.drop(remove_from_data, axis = 0, inplace = True)

data_ind = metadata.loc[metadata["sampID"].isin(data.index),"UserNumber"]
metadata.index = metadata["UserNumber"]
data.index = data_ind
data.sort_index(inplace=True)
metadata.sort_index(inplace=True)

##STUDY BIOMARKERS:
biomarker_df = pd.read_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Data/biomarkers.txt", sep = "\t")
biomarker_df.index = biomarker_df["UserNumber"]
biomarker_df.sort_index(inplace=True)
test = np.intersect1d(biomarker_df.index.values, metadata.index.values)
biomarker_df = biomarker_df.loc[test,:]
biomarker_df = biomarker_df[~biomarker_df.index.duplicated(keep='first')]



msm = metadata[metadata["all_MSMnonMSM"] == "MSM"].index.values
nonmsm = metadata[metadata["all_MSMnonMSM"] == "nonMSM"].index.values
fem = metadata[metadata["cat1"] == "FND"].index.values

metadata.loc[msm,"MSM"] = "MSM"
metadata.loc[nonmsm,"nonMSM"] = "nonMSM"
metadata.loc[fem,"FEM"] = "FEM"

otu_max_abund = data.div(data.sum(axis = 1), axis = 0).max(axis = 0)
otu_filter_out = otu_max_abund[otu_max_abund < 0.001].index

data = data.drop(otu_filter_out, axis = 1)



#%%





class HIVCohortClassification:

    def __init__(self, cohort_subset, target, norm, save, title, filename):
        self.cohort_subset = cohort_subset
        self.target = target
        self.norm = norm
        self.save = save
        self.title = title
        self.filename = filename

    def preprocess(self, X, y):
        if self.cohort_subset is not None:
            classes = y[self.cohort_subset].dropna().unique()
            data_keep =  y[y[self.cohort_subset].isin(classes)].index
            X = X[X.index.isin(data_keep)]
            y = y[y.index.isin(data_keep)]
        print self.target
        host_ids = y.index.values
        y = y[self.target]
        X = X.values
        if self.norm:
            X = np.log(X + 1.0)
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
        X, y, host_ids = shuffle(X, y, host_ids)
        if self.target == 'infstat':
            y = y.map({'uninfected': 0, 'infected': 1})
        else:
            y = y.map({'nonMSM': 0, 'MSM': 1})
        return X, y, host_ids

    def RepeatedKFoldCV(self, X, y, host_ids, n_folds, n_repeats, reg, filename):
        tprs = []                    
        aucs = []                         
        importances = []                  
        mean_fpr = np.linspace(0, 1, 101)
        predictions = pd.DataFrame([], index = host_ids, columns = range(n_repeats))
        ##Repeat Cross Validation
        for n, seed in enumerate(np.random.randint(0, high=10000, size = n_repeats)):
            cv = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
            #K fold Cross Validation
            for fold_num, (train, test) in enumerate(cv.split(X, y, host_ids)):     
                alg = LogisticRegression(solver = 'liblinear', penalty = reg, class_weight =  'balanced', random_state = 123).fit(X[train], y[train])
                probas_ = alg.predict_proba(X[test])
                y_pred = probas_[:,1]
                predictions.loc[host_ids[test],n] = y_pred
                y_true = y[test]
                imp = alg.coef_[0,:]
                importances.append(imp)
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
        importances = np.stack(importances)
        importances = pd.DataFrame(importances, columns = data.columns).mean(axis = 0).sort_values(ascending = False)
        importances.to_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Results/" + filename)
        self.plotROC(tprs, mean_fpr, aucs)
        return importances, predictions
    
    def plotROC(self, tprs, mean_fpr, aucs):
        plt.figure(1, figsize=(8, 8))
        plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle=':', lw=2, color='k', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='black',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.title)
        plt.legend(loc="lower right")
        if self.save == True:
            plt.savefig("/Users/sklarjg/Desktop/MICROBIOME/HIV/Results/" + self.filename, dpi = 300)
        plt.show()


#%%
        
##Classify MSM Status across Full dataset
MSM = HIVCohortClassification(None, "cat3", True, False, '10-Fold Cross Validation for MSM Classification', "MSMvnonMSM_ALL_lasso_10fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l1", "MSM_OTU_importances_lasso_10fold.csv")
#msm_predictions["cat3"] = metadata["cat3"]
#msm_predictions["infstat"] = metadata["infstat"]
#msm_predictions.to_csv("/Users/sklarjg/Desktop/MICROBIOME/HIV/Results/msm_predictions_lasso_10fold.csv")

##Classify HIV status across Full Dataset
HIV = HIVCohortClassification(None, "infstat", True, False, '10-Fold Cross Validation for HIV Classification', "HIVposvHIVneg_ALL_lasso_10fold.png")
X, y, host_ids = HIV.preprocess(data, metadata)
hiv_importances, hiv_predictions = HIV.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l1", "HIV_OTU_importances_lasso_10fold.csv")
#%%
##Classify MSM Status across Full dataset
MSM = HIVCohortClassification(None, "cat3", True, False, '5-Fold Cross Validation for MSM Classification', "MSMvnonMSM_ALL_ridge_5fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "MSM_OTU_importances_ridge_5fold.csv")


##Classify HIV status across Full Dataset
HIV = HIVCohortClassification(None, "infstat", True, False, '5-Fold Cross Validation for HIV Classification', "HIVposvHIVneg_ALL_ridge_5fold.png")
X, y, host_ids = HIV.preprocess(data, metadata)
hiv_importances, hiv_predictions = HIV.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "HIV_OTU_importances_ridge_5fold.csv")

#############
MSM = HIVCohortClassification("MSM", "infstat", True, False, '5-Fold Cross Validation for HIV Classification in MSM', "Subsets/HIVpos_v_HIVneg_MSM_ridge_5fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "Subsets/HIVpos_v_HIVneg_MSM.csv")

#############
MSM = HIVCohortClassification("nonMSM", "infstat", True, False, '5-Fold Cross Validation for HIV Classification in non-MSM', "Subsets/HIVpos_v_HIVneg_nonMSM_ridge_5fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "Subsets/HIVpos_v_HIVneg_nonMSM.csv")

#############
MSM = HIVCohortClassification("FEM", "infstat", True, False, '5-Fold Cross Validation for HIV Classification in Women', "Subsets/HIVpos_v_HIVneg_FEM_ridge_5fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "Subsets/HIVpos_v_HIVneg_FEM.csv")

#############
MSM = HIVCohortClassification("D_MSMnonMSM", "D_MSMnonMSM", True, False, '5-Fold Cross Validation for MSM Classification in Dutch Men', "Subsets/MSMvnonMSM_dutchmen_ridge_5fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "Subsets/MSM_v_nonMSM_dutchmen.csv")

#############
MSM = HIVCohortClassification("all_MSMnonMSM", "all_MSMnonMSM", True, False, '5-Fold Cross Validation for MSM Classification in Men', "Subsets/MSMvnonMSM_men_ridge_5fold.png")
X, y, host_ids = MSM.preprocess(data, metadata)
msm_importances, msm_predictions = MSM.RepeatedKFoldCV(X, y, host_ids, 5, 20, "l2", "Subsets/MSM_v_nonMSM_men.csv")



