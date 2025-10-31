import pandas as pd
import numpy as np
import argparse
import time
import random
from tqdm import tqdm
import statistics
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score

def parse_args():
    parser = argparse.ArgumentParser("Script to binary classify samples from gene expression data using different classifiers")
    parser.add_argument('--cl',
                        help="Chosen classifier (str)",
                        type=str,
                        dest='classifier')
    parser.add_argument('--gex',
                        help="Path to the gene expression dataframe (str)",
                        type=str,
                        dest='gene_exp')
    parser.add_argument('--cv',
                        help="Chosen cross-validation generator (str)",
                        type=str,
                        dest='cv')
    parser.add_argument('--k',
                        help="Number of folds in cross validation (int)",
                        type=str,
                        dest='estimators')
    parser.add_argument('--db',
                        help="PPI database (str)",
                        type=str,
                        dest='database')                    
    parser.add_argument('--l',
                        help="Input labels from training/testing samples (.csv/.tsv)",
                        type=str,
                        dest='labels')
    parser.add_argument('--rs',
                        help="Random state",
                        type=str,
                        dest='rs')
    parser.add_argument('--o',
                        help="Output directory",
                        type=str,
                        dest='output')  
    parser.add_argument('--val',
                        help="Path to the gene expression dataframe used for validation (str)",
                        type=str,
                        dest='val_gex')
    parser.add_argument('--vlab',
                        help="Input labels from validation samples (.tsv)",
                        type=str,
                        dest='v_labels')
    opts = parser.parse_args()
    return opts
    

def gene_exp(gene_exp_path):
    '''Parameters:
    gene_exp_path = path to the gene expression dataframe (type:str)
    Output:
    Dataframe with samples as rows and gene expression as columns
    '''
    if gene_exp_path.endswith('.csv'):
        df = pd.read_csv(gene_exp_path)
    elif gene_exp_path.endswith('.tsv'):
        df = pd.read_csv(gene_exp_path, sep = '\t')
    data_w_genes = df.T
    data_w_genes = data_w_genes.rename(columns=data_w_genes.iloc[(len(data_w_genes)-1)]).drop(['probe'])
    return data_w_genes

def model_classification(data, clf, X, y, val_data, X_val, y_val, cv, db, k, rs=None):
    if clf == 'logistic_regression':
        model = LogisticRegression(max_iter = 2000)
    if clf == 'logistic_regression_l1':
        model = LogisticRegression(max_iter = 2000, penalty = 'l1', solver='saga')
    if clf == 'random_forest':
        model = RandomForestClassifier(n_estimators = 100, criterion='entropy')
    if clf == 'xgboost':
        model = XGBClassifier(learning_rate=0.01, n_estimators=100)

    if cv == 'ss':
        cv_gen = ShuffleSplit(n_splits=int(k), test_size=0.2, random_state=rs)
    if cv == 'skf':
        cv_gen = StratifiedKFold(n_splits=int(k), shuffle = True)
    if cv == 'kf':
        cv_gen = KFold(n_splits=int(k), shuffle=True, random_state=rs)

    results = cross_validate(model,
                             X=X,
                             y=y,
                             cv=cv_gen,
                             scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                             return_train_score = True,
                             return_estimator = True)
    
    df_output = pd.DataFrame(index = data.columns)
    df_val = pd.DataFrame(columns = list(val_data.index))
    df_val_sum = pd.DataFrame(columns = ['Validation accuracy', 'Validation AUC', 'Validation F1-score', 'Validation Recall'])
    v_accs = []
    v_aucs = []
    v_f1s = []
    v_r = []

    #A dataframe with the importance per fold of every feature (gene) is generated, features as index and folds as columns
    for idx, estimator in enumerate(results['estimator']):
        if clf.startswith('logistic_regression'):
             df_output['fold'+str(idx+1)] = estimator.coef_[0]
             prediction = estimator.predict(X_val)
             proba = estimator.predict_proba(X_val)
             df_val.loc[idx] = prediction
             df_val.loc[idx+int(k)] = proba[:,1]
             v_accs.append(accuracy_score(y_val, prediction))
             v_r.append(recall_score(y_val, prediction))
             fpr, tpr, thresholds = roc_curve(y_val, proba[:,1], pos_label=1)
             v_aucs.append(auc(fpr, tpr))
             v_f1s.append(f1_score(y_val, prediction))
        if (clf == 'random_forest') | (clf == 'xgboost'):
             df_output['fold'+str(idx+1)] = estimator.feature_importances_
             prediction = estimator.predict(X_val)
             proba = estimator.predict_proba(X_val)
             df_val.loc[idx] = prediction
             df_val.loc[idx+int(k)] = proba[:,1]
             v_accs.append(accuracy_score(y_val, prediction))
             v_r.append(recall_score(y_val, prediction))
             fpr, tpr, thresholds = roc_curve(y_val, proba[:,1], pos_label=1)
             v_aucs.append(auc(fpr, tpr))
             v_f1s.append(f1_score(y_val, prediction))
             
    df_val_sum['Validation accuracy'] = v_accs
    df_val_sum['Validation AUC'] = v_aucs
    df_val_sum['Validation F1-score'] = v_f1s
    df_val_sum['Validation Recall'] = v_r

    df_output.to_csv(output+'/'+clf+'_'+cv+'_'+str(k)+'_'+db+'_feature_importance.tsv', index = True, sep ='\t')
    df_val.to_csv(output+'/'+clf+'_'+cv+'_'+str(k)+'_'+db+'_val_prediction.tsv', index = False, sep ='\t')
    df_val_sum.to_csv(output+'/'+clf+'_'+cv+'_'+str(k)+'_'+db+'_val_scores.tsv', index = False, sep ='\t')

    #A dataframe with all accuracy parameters is generated
    scores = pd.DataFrame(data={"Training AUC scores": results['train_roc_auc'],
                                "Training Accuracy scores": results['train_accuracy'],
                                "Training Precision scores": results['train_precision'],
                                "Training Recall scores": results['train_recall'],
                                "Training F1 scores": results['train_f1'],
                                "Test AUC scores": results['test_roc_auc'],
                                "Test Accuracy scores": results['test_accuracy'],
                                "Test Precision scores": results['test_precision'],
                                "Test Recall scores": results['test_recall'],
                                "Test F1 scores": results['test_f1']})
    scores.to_csv(output+'/'+clf+'_'+cv+'_'+str(k)+'_'+db+'_scores.tsv', index = False, sep ='\t')

    #A txt file with a summary (median) accuracy scores is generated
    lines = ["Median Training AUC: "+str(round(statistics.median(results['train_roc_auc']),3)),
             "Median Training Accuracy: "+str(round(statistics.median(results['train_accuracy']),3)),
             "Median Training Precision: "+str(round(statistics.median(results['train_precision']),3)),
             "Median Training Recall: "+str(round(statistics.median(results['train_recall']),3)),
             "Median Training F1 Score: "+str(round(statistics.median(results['train_f1']),3)),
             "Median Test AUC: "+str(round(statistics.median(results['test_roc_auc']),3)),
             "Median Test Accuracy: "+str(round(statistics.median(results['test_accuracy']),3)),
             "Median Test Precision: "+str(round(statistics.median(results['test_precision']),3)),
             "Median Test Recall: "+str(round(statistics.median(results['test_recall']),3)),
             "Median Test F1 Score: "+str(round(statistics.median(results['test_f1']),3))]
    
    with open(output+'/'+clf+'_'+cv+'_'+str(k)+'_'+db+'_summary.txt', 'w') as f:
              f.write('\n'.join(lines))
              
start = time.time() 

args = parse_args()
gex = args.gene_exp
clf = args.classifier
cv_gen = args.cv
k = args.estimators
db = args.database
path_to_labels = args.labels
random_state = args.rs
output = args.output
gex_new = args.val_gex
val_lab = args.v_labels

if path_to_labels.endswith('.csv'):
    labels = pd.read_csv(path_to_labels)
elif path_to_labels.endswith('.tsv'):
    labels = pd.read_csv(path_to_labels, sep = '\t')

if val_lab.endswith('.csv'):
    v_labels = pd.read_csv(val_lab)
elif val_lab.endswith('.tsv'):
    v_labels = pd.read_csv(val_lab, sep = '\t')
    
data = gene_exp(gex)
X, y = data.to_numpy(), np.array(labels.iloc[0])

val_data = gene_exp(gex_new)
X_val, y_val = val_data.to_numpy(), np.array(v_labels.iloc[0])

model_classification(data, clf, X, y, val_data, X_val, y_val, cv_gen, db, k, random_state)
end = time.time()
print('Total time: '+str(round(((end-start)/60),2))+' min')