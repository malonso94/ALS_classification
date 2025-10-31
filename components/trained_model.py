from tensorflow.python.ops import gen_nn_ops
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy
from lib import graph
import time

class TrainedModel:

    def __init__(self, model, samples):
        """
        Initialization of internals for relevance computations.
        :param model: gcnn model to LRP procedure is applied on
        :param samples: samples to calculate relevance on, num of samples <= models batch size
        """

        weights = model.get_weights()
        self.activations = model.activations

        self.model = model
        self.model.graph._unsafe_unfinalize()  # the computational graph of the model will be modified
        self.samples = samples
        self.X = self.activations[0]  # getting the first

        # self.y = self.activations.pop(0)
        # I am getting the activation of the first, but not useful here, self.y will be assigned a placeholder

        self.ph_dropout = model.ph_dropout
        self.batch_size = self.X.shape[0]

        # with self.model.graph.as_default():
        #     self.y = tf.placeholder(tf.float32, (self.batch_size, labels.shape[1]), 'labels_hot_encoded')

        self.act_weights = {}  # example in this dictionary "conv1": [weights, bias]

        for act in self.activations[1:]:
            w_and_b = []  # 2 element list of weight and bias of one layer.
            name = act.name.split('/')
            # print(name)
            for wt in weights:
                # print(wt.name)
                if name[0] == wt.name.split('/')[0]:
                    w_and_b.append(wt)
            if w_and_b and (name[0] not in self.act_weights):
                self.act_weights[name[0]] = w_and_b

    def run_tf_tensor(self, tensor, samples):
        """
        Runs computational graph to get the Numpy values of the tensor.
        :param tensor:
        :param samples:
        :return:the value of a specific tensor based on the
        """
        # TODO: rewrite it as a concatenation of all the batches per each activation
        data = np.expand_dims(samples, axis=2)
        size = data.shape[0]
        # activations = [] # np.empty([size, len(self.activations) - 2]) # TODO: adjust the number of tensors
        # print(R.shape)
        # session =
        tnsr = None
        tensors = []
        with self.model._get_session() as sess:
            for begin in range(0, size, self.batch_size):
                end = begin + self.batch_size
                end = min([end, size])
                batch_data = np.zeros((self.batch_size, data.shape[1], 1))
                tmp_data = data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end - begin] = tmp_data
                if end == size:
                    print(batch_data[-1,:])
                print(f"begin: {begin}, end: {end}, batch_data : {batch_data.shape}, tmp_data: {tmp_data.shape}")
                feed_dict = {self.X: batch_data, #self.y: self.labels,
                             self.ph_dropout: 1}
                batch_tnsr = sess.run(tensor, feed_dict)

                # if tnsr is None:
                #     tnsr =  batch_tnsr
                # else:
                #     tnsr = tf.concat([tnsr,batch_tnsr],0)
                tensors.append(batch_tnsr)
                # for ba in batch_activations:
                #     print("The whole batch:")
                #     print("ba:", ba.shape)
                #     print("ba, Gb:", ba.nbytes / (1024*1024*1024))
                #     activations.append(ba)

        return tensors
    