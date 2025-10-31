#!python

"""
Script for validating the GCNN model (predict the labels of the validation dataset)
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import argparse
import re
from statistics import mean 
from scipy.sparse import csr_matrix
from components import data_handling
from lib import coarsening

def parse_args():

    parser = argparse.ArgumentParser("Script for validating the GCNN model")

    parser.add_argument('--gx',
                        help="Path to the gene expression data of the validation set",
                        type=str,
                        dest='gex')
    parser.add_argument('--adm',
                        help="Path to the adyacency matrix (should be the same as the one used for training)",
                        type=str,
                        dest='ady_mat')
    parser.add_argument('--l',
                        help="Path to the labels from patients",
                        type=str,
                        dest='labels')
    parser.add_argument('--p',
                        help="Params (name of the folder where the model is saved)",
                        type=str,
                        dest='params')
    parser.add_argument('--o',
                        help="Output directory",
                        type=str,
                        dest='output')
    parser.add_argument('--n',
                        help="Run number",
                        type=int,
                        dest='num')
    parser.add_argument('--db',
                        help="PPI database",
                        type=str,
                        dest='database')

    opts = parser.parse_args()
    return opts

def val_classification(num, params, lvl, gex_path, adj_path, labels_path, out_dir, db):
    labels = pd.read_csv(labels_path, sep = '\t')
    
    DP = data_handling.DataPreprocessor(path_to_feature_values=gex_path,
                                        path_to_feature_graph=adj_path,
                                        path_to_labels=labels_path)
    
    X = DP.get_feature_values_as_np_array()  # Gene expression
    A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # Compressed sparse row (CSR) of the adjacency matrix of the PPI network
    y = DP.get_labels_as_np_array()  # Labels
    
    print("GEx data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    print("PPI network adjacency matrix, A shape: ", A.shape)
    
    X_new = X - np.min(X)
    
    _, perm = coarsening.coarsen(A, levels=lvl, self_connections=False)
    X_new = coarsening.perm_data(X_new, perm)
    print('Input shape:', X_new.shape)

    for n in num:
        checkpoint_dir = '/path_to_output_folder/'+db+'/r'+str(n)+'_'+params+'/checkpoints/GE' # Adjust the path as needed
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
        # Create a session & restore the latest checkpoint
        sess = tf.Session()
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        saver.restore(sess, latest_checkpoint)
        
        input_tensor = sess.graph.get_tensor_by_name('inputs/data:0')
        output_tensor = sess.graph.get_tensor_by_name('prediction/Softmax:0')
        dropout_tensor = sess.graph.get_tensor_by_name('inputs/dropout:0')
    
        if params.startswith('k0.1'):
            batch_size = 75 # Adjust batch size as needed depending on the dataset size
        
        batches = [X_new[i:i+batch_size, :] for i in range(0, X_new.shape[0], batch_size)]
    
        probs = []
        preds = []
        
        for batch in batches:
            zero_array = np.zeros((batch_size, X_new.shape[1]))
            zero_array[:batch.shape[0]] = batch
            
            # 'new_samples' are the samples that you want to predict the labels for (same size as input_tensor)
            probabilities = sess.run(output_tensor, feed_dict={input_tensor: zero_array, dropout_tensor: 1})
            predictions = probabilities.round().astype(int)
        
            a = [i[1] for i in probabilities[:batch.shape[0],:batch.shape[1]]]
            b = [i[1] for i in predictions[:batch.shape[0],:batch.shape[1]]]
            
            probs.extend(a)
            preds.extend(b)
            
        df_results = pd.DataFrame(columns = ['patient_id', 'label', 'prediction'])
        df_results['patient_id'] = list(labels.columns)
        df_results['label'] = y
        df_results['prediction'] = preds
        df_results['probabilities'] = probs
        df_results.to_csv(out_dir+'pred_validation.tsv', sep = '\t', index = False)

args = parse_args()
gex_path = args.gex
adj_mat = args.ady_mat
labels_path = args.labels
params = args.params
out_dir = args.output
db = args.database
num = [args.num]

regex = '_l\d_'
level = re.findall(regex, params)

for l in level:
    lvl = int(l.replace('_l', '').replace('_', ''))

val_classification(num, params, lvl, gex_path, adj_mat, labels_path, out_dir, db)