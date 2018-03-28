#--*-- coding:utf-8 --*--
import tensorflow as tf
from cnn import CNN
from utils import cos_sim, max_pooling, avg_pooling, cal_loss_and_acc, all_pool
import tensorflow.contrib.layers as layers
from bilstm import BILSTM


class CNN_QA(object):
    def __init__(self, batch_size, seq_len, embeddings, embedding_size, filter_size, num_filters, num_features, num_layers, rnn_size=100, unknown_id=7447, num_classes=2, l2_reg_lambda=4e-4, model_type= "ABCNN3", adjust_weight=False,label_weight=[],is_training=True):
        # define input variable
        self.batch_size = batch_size
        self.seq_len = seq_len 
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_features = num_features
        self.num_layers = num_layers 
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.model_type = model_type
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.is_training = is_training
        self.rnn_size = rnn_size

        self.ori_input_quests = tf.placeholder(tf.int32, shape=[None, self.seq_len], name="ori_input")
        self.cand_input_quests = tf.placeholder(tf.int32, shape=[None, self.seq_len], name="cand_input")
        #self.ori_input_quests_var = tf.placeholder(tf.int32, shape=[None, None], name="ori_input_var")
        #self.cand_input_quests_var = tf.placeholder(tf.int32, shape=[None, None], name="cand_input_var")
        #self.ori_input_quests_var = self.ori_input_quests
        #self.cand_input_quests_var = self.cand_input_quests
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        
        self.new_lr = tf.placeholder(tf.float32, shape=[],name="new_learning_rate")
        self.lr = tf.Variable(0.0,trainable=False)
        self._lr_update = tf.assign(self.lr, self.new_lr)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")

        ori_quests =tf.nn.embedding_lookup(W, self.ori_input_quests)
        cand_quests =tf.nn.embedding_lookup(W, self.cand_input_quests)
        #ori_quests_var =tf.nn.embedding_lookup(W, self.ori_input_quests_var)
        #cand_quests_var =tf.nn.embedding_lookup(W, self.cand_input_quests_var)

        #shape [batch_size, embedding_size, seq_len, 1]

        #shape [batch_size, embedding_size]
        #LO_0 = all_pool("input-left", x1_expanded, self.seq_len, self.filter_size, self.num_filters, self.embedding_size)
        #RO_0 = all_pool("input-right", x2_expanded, self.seq_len, self.filter_size, self.num_filters, self.embedding_size)

        # LI_1, RI_1 shape [batch, num_filters, seq_len, 1]
        # LO_1, RO_1 shape [batch, num_filters]
        #x1_expanded = tf.expand_dims(ori_quests, -1)
        #x2_expanded = tf.expand_dims(cand_quests, -1)
        #LO_1, RO_1 = CNN_layer("CNN-1", x1_expanded, x2_expanded, self.seq_len, self.embedding_size, self.num_filters, self.filter_size, self.l2_reg_lambda, self.model_type) 
        #with tf.variable_scope("cnn", reuse=None) as scope:
        #    LO_1 = CNN(x1_expanded, self.seq_len, self.embedding_size, self.filter_size, self.num_filters) 
        #with tf.variable_scope("cnn", reuse=True) as scope:
        #    RO_1 = CNN(x2_expanded, self.seq_len, self.embedding_size, self.filter_size, self.num_filters) 

        with tf.variable_scope("LSTM_scope", reuse=None):
            ori_q = BILSTM(ori_quests, self.rnn_size)
            LO_1 = max_pooling(ori_q)
        with tf.variable_scope("LSTM_scope", reuse=True):
            cand_a = BILSTM(cand_quests, self.rnn_size)
            RO_1 = max_pooling(cand_a)
        #self.sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)]
        #self.sims = [cos_sim(LO_1, RO_1), cos_sim(LO_2, RO_2)]
        self.sims = [cos_sim(LO_1, RO_1)]

        if self.num_layers > 1:
            with tf.variable_scope("cnn", reuse=None) as scope:
                LO_2 = CNN(tf.expand_dims(ori_q, -1), self.seq_len, self.rnn_size * 2, self.filter_size, self.num_filters) 
            with tf.variable_scope("cnn", reuse=True) as scope:
                RO_2 = CNN(tf.expand_dims(cand_a, -1), self.seq_len, self.rnn_size * 2, self.filter_size, self.num_filters) 
            self.sims.append(cos_sim(LO_2, RO_2))

        with tf.variable_scope("output_layer") as scope:
            self.output_features = tf.concat(1, [self.features, tf.pack(self.sims, axis=1)], name="output_features")
        self.lstm_features = tf.concat(1, [LO_2, RO_2])
        #self.lstm_features = self.output_features

        self.num_classes = 1
        with tf.variable_scope("fully_connected"):
            #feature_len = int(self.output_features.get_shape()[1])
            feature_len = int(self.lstm_features.get_shape()[1])
            softmax_w = tf.get_variable("softmax_w", initializer=tf.truncated_normal([feature_len, self.num_classes], stddev=0.1))
            softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(0., shape=[self.num_classes]))
            #self.estimation = tf.matmul(self.output_features, softmax_w) + softmax_b
            self.estimation = tf.nn.sigmoid(tf.matmul(self.lstm_features, softmax_w) + softmax_b)
            #self.output_features = tf.concat(1, [self.features, tf.nn.softmax(self.estimation)])
            #self.output_features = tf.concat(1, [self.features, self.estimation])

        with tf.name_scope("loss"):
            #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.estimation, self.labels)
            #self.cost = tf.reduce_mean(self.loss)
            self.cost = tf.reduce_mean(tf.square(tf.cast(self.labels, "float32") - self.estimation))


    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
