# AETSI
This repository contains the code used to produce the results in the paper "Approximating the Hotelling Observer with Autoencoder-Learned Efficient Channels for Binary Signal Detection Tasks".

Files for each of the primary experiments are contained in their own folder. Within each category, there are folders that contain the data, models, and scripts.  

The data and models folders are empty in this distribution, but are kept to demonstrate the directory structure. Contents of the data and model directories can be found on the Havard database associated with this project:  

Granstedt, Jason, 2023, "Replication Data for: Approximating the Hotelling Observer with Autoencoder-Learned Efficient Channels for Binary Signal Detection Tasks", https://doi.org/10.7910/DVN/BQNOMZ, Harvard Dataverse

The scripts folder contains the scripts employed to generate the statistics in the paper. The included code contains the python training scripts, shell scripts for sweeping the relevant parameters of the models, and MATLAB scripts for generating statistics. The scripts folders contain a subfolder named "VICTRE", which contains the code for the filtered channel observer. This code was taken from https://github.com/DIDSR/VICTRE_MO.  

The model training scripts are designed to be run on Tensorflow 1.14.

The following is the pipeline to train models and generate test statistics:

1. Edit one of the four shell scripts corresponding to the desired signal/background combination to train the desired models.  
2. Run gridsearch.m to obtain the best values for the desired signal/background combination.  The type variable at the top of the file loads data for one of the signal/background combinations, which can be seen in the data_path immediately under each case. For example, data_path = '../Data/Phantom/spicmass/ indicates that the spiculated mass signal for the VICTRE numerical breast phantom background will be loaded.
3. Record the best parameters for each model from the gridsearch.m run.  
4. Insert the best parameters for each model into the corresponding type entry in gen_stats_tmi.m
5. Run gen_stats_tmi.m, making sure to change the type entry at the top to the same one used in step 2.  
6. The test statistics will be written to a corresponding test statistics folder. Fit these test statistics to replicate the AUC values presented in the paper.

For the generalization studies, use the following pipeline instead:  
1. Run each of the lumpy_channelsweep shell scripts.
2. Run cross_analysis_full. This combines steps 2-5 in the previous pipeline, writing the test statistic files.
3. Fit the test statistics to replicate the results presented in the paper.