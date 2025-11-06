#!python

"""
Relevances obtained for testing patients are written into the file "relevances_rendered_class.csv".
The file "predicted_concordance.csv" contains a table showing which patients were predicted correctly.
"""
import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from components import data_handling, glrp_scipy, trained_model
import argparse
from lib import models, graph, coarsening

import time

def parse_args():

    parser = argparse.ArgumentParser("Script to classify patients based on PPI network with gene expression data")

    parser.add_argument('--gx',
                        help="input gene expression file (.tsv)",
                        type=str,
                        dest='gx_file')
    parser.add_argument('--adm',
                        help="input adyacency matrix file (.tsv)",
                        type=str,
                        dest='ady_mat')
    parser.add_argument('--l',
                        help="input labels from patients (.csv)",
                        type=str,
                        dest='labels')
    parser.add_argument('--o',
                        help="output directory",
                        type=str,
                        dest='output')
    parser.add_argument('--k',
                        help="kfold (as fraction)",
                        type=float,
                        dest='kfold')
    parser.add_argument('--ep',
                        help="n° of epochs",
                        type=int,
                        dest='num_epochs')
    parser.add_argument('--ef',
                        help="evaluation frequency",
                        type=float,
                        dest='eval_frequency')
    parser.add_argument('--pl',
                        help="type of pooling",
                        type=str,
                        dest='pool')
    parser.add_argument('--d',
                        help="dropout rate",
                        type=float,
                        dest='dropout')
    parser.add_argument('--lv',
                        help="levels",
                        type=int,
                        dest='levels')
    parser.add_argument('--po',
                        help="polynomial order, K-hop neighborhood around a node",
                        type=int,
                        dest='pol_ord')
    parser.add_argument('--rs',
                        help="random state",
                        type=int,
                        dest='rnd_st')
    parser.add_argument('--lr',
                        help="learning rate",
                        type=float,
                        dest='l_rate')

    opts = parser.parse_args()
    return opts

if __name__ == "__main__":

    args = parse_args()
    path_to_feature_val = args.gx_file
    path_to_feature_graph = args.ady_mat
    path_to_labels = args.labels
    dir_to_save = args.output
    test_size = args.kfold 
    num_epochs = args.num_epochs 
    eval_freq = args.eval_frequency 
    pool = args.pool
    dropout = args.dropout
    lvs = args.levels 
    pol_ord = args.pol_ord 
    rndm_state = args.rnd_st
    learning_rate = args.l_rate

    checkpoint_path = dir_to_save+'checkpoints/'

    DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,
                                    path_to_labels=path_to_labels)
    X = DP.get_feature_values_as_np_array()  # gene expression
    A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # compressed sparse row (CSR) of the adjacency matrix of the PPI network
    y = DP.get_labels_as_np_array()  # labels

    print("GE data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    print("PPI network adjacency matrix, A shape: ", A.shape)

    X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                        stratify=y, random_state=rndm_state)

    # Need to know which patients got into train and test subsets
    _, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=test_size,
                                                                        stratify=y, random_state=rndm_state)

    # Dataframe with test patients and corresponding ground truth labels
    patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})

    # Making data lying in the interval [0, 8.35]
    X_train = X_train_unnorm - np.min(X)
    X_test = X_test_unnorm - np.min(X)

    print("X_train max", np.max(X_train))
    print("X_train min", np.min(X_train))
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train, shape: ", y_train.shape)
    print("y_test, shape: ", y_test.shape)

    # Coarsening the PPI graph to perform pooling in the model
    graphs, perm = coarsening.coarsen(A, levels=lvs, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    X_train = coarsening.perm_data(X_train, perm)
    X_test = coarsening.perm_data(X_test, perm)

    n_train = X_train.shape[0]

    params = dict()
    params['dir_name']       = 'GE'
    params['num_epochs']     = num_epochs
    params['batch_size']     = int(np.ceil(y.shape[0]*test_size))
    params['eval_frequency'] = eval_freq

    # Building blocks.
    params['filter']         = 'chebyshev5'
    params['brelu']          = 'b1relu'
    params['pool']           = pool

    # Number of classes.
    C = y.max() + 1
    assert C == np.unique(y).size

    # Architecture.
    params['F']              = [32 for i in range(0, lvs)]  # Number of graph convolutional filters.
    params['K']              = [pol_ord for i in range(0, lvs)]  # Polynomial orders.
    params['p']              = [2 for i in range(0, lvs)]    # Pooling sizes.
    params['M']              = [512, 128, C]  # Output dimensionality of fully connected layer L, F, K, p, M
    params['dropout']        = dropout # Originally was 1 (without dropout)
    params['learning_rate']  = learning_rate # Originally was 0.001
    params['decay_rate']     = 0.95
    params['momentum']       = 0 #0 for adam, !=0 for momentum
    params['decay_steps']    = n_train / params['batch_size']
    params['checkpoint_path']    = checkpoint_path

    model = models.cgcnn(L, **params)
    
    # TRAINING.
    # In case the trained model is saved: simply comment the three lines below to run glrp again.
    start = time.time()
    accuracy, loss, t_step, trained_losses = model.fit(X_train, y_train, X_test, y_test, dir_to_save)
    end = time.time()

    # Saving the probabilities and the summary
    probas_V = model.get_probabilities(X_test)
    probas_T = model.get_probabilities(X_train)
    probas_df_T = pd.DataFrame(np.concatenate([probas_T, y_train.reshape(-1, 1)], axis = 1), columns = ['prob0', 'prob1', 'label'])
    probas_df_V = pd.DataFrame(np.concatenate([probas_V, y_test.reshape(-1, 1)], axis = 1), columns = ['prob0', 'prob1', 'label'])
    probas_df_T.to_csv(path_or_buf = dir_to_save + "/probabilities_train.csv", index=False)
    probas_df_V.to_csv(path_or_buf = dir_to_save + "/probabilities_test.csv", index=False)
    labels_by_network = np.argmax(probas_V, axis=1)

    fpr, tpr, _ = roc_curve(y_test, probas_V[:, 1])
    roc_auc = auc(fpr, tpr)
    f1 = 100 * f1_score(y_test, labels_by_network, average='weighted')
    acc = accuracy_score(y_test, labels_by_network)
    print("\n\tTest AUC:", roc_auc) # np.argmax(y_preds, axis=2)[:, 0] fot categorical
    print("\tTest F1 weighted: ", f1)
    print("\tTest Accuraccy:", acc, "\n")

    with open(dir_to_save+"/summary.txt", 'w') as f:
        f.write("Test AUC: "+str(roc_auc)+"\n")
        f.write("Test F1 weighted: "+str(f1)+"\n")
        f.write("Test Accuraccy: "+str(acc))

    # Creating hot-encoded labels for GLRP
    I = np.eye(C)
    tmp = I[labels_by_network]
    labels_hot_encoded = np.ones((model.batch_size, C))
    labels_hot_encoded[0:tmp.shape[0], 0:tmp.shape[1]] = tmp
    print("labels_hot_encoded.shape", labels_hot_encoded.shape)

    print("labels_by_network type", labels_by_network.dtype)
    print("y_test type", y_test.dtype)
    concordance = y_test == labels_by_network
    concordance = concordance.astype(int)
    out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),
                                      columns=["Predicted", "Concordance"])
    concordance_df = patient_ind_test_df.join(out_labels_conc_df)
    concordance_df.to_csv(path_or_buf = dir_to_save + "/predicted_concordance.csv", index=False)

    # CALCULATION OF RELEVANCES
    # CAN TAKE QUITE SOME TIME
    glrp = glrp_scipy.GraphLayerwiseRelevancePropagation(model, X_test, labels_hot_encoded)
    rel = glrp.get_relevances()[-1][:X_test.shape[0], :]
    rel = coarsening.perm_data_back(rel, perm, X.shape[1])
    rel_df = pd.DataFrame(rel, columns=DP.feature_names)
    rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
    rel_df.to_csv(path_or_buf = dir_to_save + "/relevances_rendered_class.csv", index=False)

    # Saving the tensors
    my_model = trained_model.TrainedModel(model, X_train)
    my_tensors = my_model.run_tf_tensor(my_model.activations[-2], X_train)
    my_modelV = trained_model.TrainedModel(model, X_test)
    my_tensorsV = my_model.run_tf_tensor(my_model.activations[-2], X_test)

    tensors_train = pd.DataFrame()
    for t in my_tensors:
        tensors_train = pd.concat([tensors_train, pd.DataFrame(t)], axis=0)
    tensors_train = tensors_train.T
    tensors_train = tensors_train.iloc[:, 0:len(patient_indexes_train)]
    tensors_train = tensors_train.set_axis(patient_indexes_train, axis=1)

    tensors_test = pd.DataFrame(my_tensorsV[0]).T
    tensors_test = tensors_test.set_axis(patient_indexes_test, axis=1)

    tensors_final = pd.concat([tensors_train, tensors_test], axis=1)
    tensors_final.to_csv(path_or_buf = dir_to_save + "/tensors.csv", index=False)