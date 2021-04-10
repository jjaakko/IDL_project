# Multi-label classification on the Reuters RCV1 corpus

In this project we experiment on two different deep learning architectures in an effort to do multi-label classification on the Reuters corpus. 
We built two models to tackle the classification challenge. One of these was an LSTM model and the other a BERT model.

## Structure of this repository

The original data was given as a collection of .xml files. These files have been parsed and preprcosessed with the tools found in the folder 'preprocessing'. 
Resulting .csv files are stored in the folder 'data'. All the scripts needed to build, train and evaluate the models can be found in the folders 'LSTM' and 'BERT'.
