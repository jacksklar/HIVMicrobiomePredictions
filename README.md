# HIVMicrobiomePredictions
Detecting significant bacterial strains associated with host-lifestyle factors using logistic regression 


Study cohort consisted of 160 individuals controlled by sexual orientation and HIV infection status, currently the largest of its kinds and balanced for both features. Logisic regression was trained on log-transformed-standardized Bacterial OTU count data in order to assess the significance of each feature in modifying the host's microbiome community. From the model, feature importance was used to assess the effect that each feature had on various taxonomy of bacteria, which was compared to previous studies that did not have as strong of a cohort in terms of size and balance. 

Due to the nature of the dimension of the OTU count data: small sample size (160), and far larger feature-size (4460), first the feature seletion was performed by removing all OTUs that had less than 0.1% maximum relative abundance across all samples in the dataset. Aditionally l1 and l2 regularization were tested seperately in order to reduce the likelihood of overfitting, which was a significant consern considering dimensions of the data. 

Due to the small number of samples in the constext of training a model, 20-repeat 5-fold Cross Validation was performed in order to estimate how the model would perform given a large set of samples. The Area under the Curve (AUC) of the ROC curve, averaged across all repeated models, was used to determine how significant any such host-feature dependent microbiome signature was. 

It would not ever be advised to use such models as a predictive tool for these features, as the nature of human microbiome data is extremely noisy, sparse, and most importantly, is very dependent on host geographical location and diet, which were accounted for in the construction of this cohort, but would not be gaurenteed for any new test samples. 

Additional, more conventional, statistical analysis was performed to look at the abundance, of each OTU and associated families of bacteria associated with each feature as well as within subgroups, such as men, and women. 
