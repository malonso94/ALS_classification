# MIT License
#
# Copyright (c) 2020 Hryhorii Chereda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
from numpy import uint8, float32, float64, std, mean, min, max, random, identity, array, argsort
from numpy import linalg as LA
from sklearn.preprocessing import normalize


class DataPreprocessor:
    """Performs converting of the dataset to NumPy format for feeding into the machine learning algorithm."""

    _float_precision = float64  # 64

    def __init__(self, path_to_feature_values, path_to_labels, path_to_feature_graph=None):
        self.path_feat_val = path_to_feature_values #gene expression matrix
        self.path_feat_graph = path_to_feature_graph # adjacency matrix
        self.path_to_labels = path_to_labels #labels/classification

        #Reading and transposing the GEX dataframe: samples as index and gene names as columns
        if self.path_feat_val.endswith('.tsv'):
            self.feature_values = pd.read_csv(self.path_feat_val, sep = '\t')
        elif self.path_feat_val.endswith('.csv'):
            self.feature_values = pd.read_csv(self.path_feat_val)
        else:
            print("Can't read gene expression matrix")
            raise

        cols = self.feature_values.columns.tolist()
        self.feature_names = list(self.feature_values[cols[-1]])
        self.feature_values = self.feature_values[cols[:-1]]
        self.feature_values = self.feature_values.transpose()
        self.feature_values.columns = self.feature_names

        #Reading the adjacency matrix
        if (self.path_feat_graph is not None) and (self.path_feat_graph.endswith('.tsv')):
            self.adj_feature_graph = pd.read_csv(path_to_feature_graph, sep = '\t')
        elif (self.path_feat_graph is not None) and (self.path_feat_graph.endswith('.csv')):
            self.adj_feature_graph = pd.read_csv(path_to_feature_graph)
        else:
            self.adj_feature_graph = None
            print("Can't read adjacency matrix")

        #Reading the labels dataframe, they are in a row
        if (self.path_to_labels is not None) and (self.path_to_labels.endswith('.tsv')):
            self.labels = pd.read_csv(path_to_labels, sep = '\t')
        elif (self.path_to_labels is not None) and (self.path_to_labels.endswith('.csv')):
            self.labels = pd.read_csv(path_to_labels)
        else:
            self.path_to_labels = None
            print("Can't read labels")

        #Check if n° of patients in GEX df is the same as the n° of labels
        if self.feature_values.shape[0] != self.labels.shape[1]:
            print("The number of patients %d is not equal to the number of the labels %d" % (
            self.feature_values.shape[0], self.labels.shape[1]))
            raise

    def get_feature_values_as_np_array(self, columns=None):
        """Extracts values of the GEX dataframe as array (float64)"""
        if not columns:
            return self.feature_values.values.astype(self._float_precision)  # as_matrix()
        else:
            return self.feature_values[columns].values.astype(self._float_precision)

    def get_labels_as_np_array(self):
        """Extracts values the labels dataframe as array (int)"""
        return self.labels.values[0].astype(int)  # [0] because wrapped in an additional dimension

    def get_adj_feature_graph_as_np_array(self):
        """Extracts values of the adjacency matrix as array (float64)"""
        if self.adj_feature_graph is not None:
            return self.adj_feature_graph.values.astype(self._float_precision)
        else:
            print("The adjacency matrix of the graph was not provided by the user")
            raise

    def get_data_frame_for_mRMR_method(self):
        """
        Returns dataframe of gene expression values and labels.
        The first row: "class" "gen1" "gen2", etc.
        The second row: "label" "gex gen1" “gex gen2", etc.
        Index is not the sample id, it's a regular index."
        """
        values = self.get_feature_values_as_np_array()  # *10e+7
        print(max(values))
        df = pd.DataFrame(data=values)  # .astype(int))
        feat_list = list(['class'])
        feat_list.extend(self.feature_names)
        df[len(df.columns)] = self.get_labels_as_np_array()  # add labels to the right last side
        df = df[[len(df.columns) - 1] + list(range(0, len(df.columns) - 1))]  # put the labels into the left side
        df.columns = feat_list  # keeping order of columns for the next concatenation
        return df

    @staticmethod
    def normalize_data_0_1(X, eps=5e-7):
        """
        Normalize in 0_1 interval, each column of X is a feature, each row is a sample.
        Returns three variables:
        X normalized,
        array of min
        array of max values.
        """
        column_max = max(X, axis=0).astype(float64)
        column_min = min(X, axis=0).astype(float64)
        non_zero_ind = column_max > eps
        column_min = column_min[non_zero_ind]
        column_max = column_max[non_zero_ind]
        X_norm = X[:, non_zero_ind]
        X_norm = (X_norm - column_min) / (column_max - column_min)
        return X_norm, column_min, column_max, non_zero_ind

    @staticmethod
    def scale_data_0_1(X_val, column_min, column_max, non_zero_ind):
        """
        Scaling the validation data according to the min and max of features in the training data.
        Returns:
        X_val scaled.
        """
        return (X_val[:, non_zero_ind] - column_min) / (column_max - column_min)

    @staticmethod
    def normalize_data(X, eps=5e-7):
        """
        Z score calculation, each column of X is a feature, each row is a sample.
        Returns three variables:
        X normalized,
        array of mean values
        array of std values.
        """
        column_std = std(X, axis=0, ddof=1).astype(float64)  # along rows

        print("column_std.shape:", column_std.shape)
        print("column_std, num of el less than 1:", sum(column_std < 1))
        column_mean = mean(X, axis=0).astype(float64)
        non_zero_ind = column_std > eps
        column_std = column_std[non_zero_ind]
        column_mean = column_mean[non_zero_ind]
        X_norm = X[:, non_zero_ind]
        X_norm = (X_norm - column_mean) / column_std
        return X_norm, column_mean, column_std, non_zero_ind

    @staticmethod
    def scale_data(X_val, column_mean, column_std, non_zero_ind):
        """
        Scaling the validation data according to the mean and std of features in the training data.
        Returns:
        X_val scaled.
        """
        return (X_val[:, non_zero_ind] - column_mean) / column_std

    @staticmethod
    def generate_Q(transition_matrix, q_power):
        """Generate Q^{q_power} for n variables."""
        n = transition_matrix.shape[0]
        q_tmp = tr_0 = identity(n)
        for k in range(1, q_power + 1):
            q_tmp += LA.matrix_power(transition_matrix, k)
        return array(normalize(q_tmp, norm='l1', axis=1), dtype='float64')  # float 32 works inappropriately
