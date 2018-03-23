# coding=utf-8

import logging
import datetime
import time
import tensorflow as tf
import operator
import numpy as np
import os

from sklearn import linear_model
from sklearn.externals import joblib

from data_helper import load_train_data, load_test_data, load_embedding, batch_iter, cal_basic_feature, gen_neg_quest, cal_word2vec_sim, cal_Lcs
from polymerization import CNN_QA
from utils import build_path
from preprocess import Word2Vec, MSRP, WikiQA


#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../DATA/train.txt", "train corpus file")
tf.flags.DEFINE_string("tfidf_file", "../DATA/tfidf.txt", "train corpus file")
tf.flags.DEFINE_string("embedding_file", "../DATA/vectors.txt", "embedding file")
tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_integer("num_filters", 100, "embedding size")
tf.flags.DEFINE_string("filter_size", "1,2,3,4", "embedding size")
tf.flags.DEFINE_float("dropout", 0.5, "the proportion of dropout")
tf.flags.DEFINE_float("lr", 0.1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 128, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 50, "epoches")
tf.flags.DEFINE_integer("evaluate_every", 1000, "run evaluation")
tf.flags.DEFINE_integer("quest_len", 20, "embedding size")
tf.flags.DEFINE_integer("num_layers", 2, "embedding size")
tf.flags.DEFINE_string("out_dir", "save/", "output directory")
tf.flags.DEFINE_string("modelType", "BCNN", "output directory")
tf.flags.DEFINE_string("data_type", "JIMI", "output directory")
tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.8, "use memory rate")

FLAGS = tf.flags.FLAGS
filter_size = [int(each) for each in FLAGS.filter_size.split(",")]
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log", mode="w")
fh.setLevel(logging.INFO)
logger.addHandler(fh)
#----------------------------- define a logger end ----------------------------------

#------------------------------------load data -------------------------------
embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)
ori_quests, cand_quests, labels, ori_quests_var, cand_quests_var, total_quests = load_train_data(FLAGS.train_file, word2idx, FLAGS.quest_len, FLAGS.quest_len)
unknown_id = word2idx.get("UNKNOWN", 0)
features = cal_basic_feature(ori_quests_var, cand_quests_var, total_quests, embedding, word2idx, saveTfidf=True, tfidfFile=FLAGS.tfidf_file)
#cal_word2vec_sim(ori_quests_var, cand_quests_var, embedding, features)
#cal_Lcs(ori_quests_var, cand_quests_var, features)
num_feature = len(features[0])
#----------------------------------- load data end ----------------------

#----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, features_batch, t_labels, ori_quests_var_batch, cand_quests_var_batch, cnn, dropout=1.):
    start_time = time.time()
    feed_dict = {
        cnn.ori_input_quests : ori_batch,
        cnn.cand_input_quests : cand_batch, 
        cnn.features: features_batch,
        cnn.labels : t_labels,
        cnn.keep_prob : dropout
    }

    _, step, cur_lost, output_features, estimation, sims, features = sess.run([train_op, global_step, cnn.cost, cnn.output_features, cnn.estimation, cnn.sims, cnn.features], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    time_elapsed = time.time() - start_time
    logger.info("%s: step %s, loss %s, %6.7f secs/batch"%(time_str, step, cur_lost, time_elapsed))
    return output_features

#---------------------------------- execute train model end --------------------------------------

#----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:3"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cnn = CNN_QA(FLAGS.batch_size, FLAGS.quest_len, embedding, FLAGS.embedding_size, filter_size, FLAGS.num_filters, num_feature, FLAGS.num_layers, unknown_id=unknown_id, model_type=FLAGS.modelType)
            global_step = tf.Variable(0, name="globle_step",trainable=False)
            #add checkpoint
            saver = tf.train.Saver(tf.all_variables())
            checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            optimizer = tf.train.GradientDescentOptimizer(1e-3)
            #train_op = tf.train.AdamOptimizer(1e-4).minimize(cnn.cost)
            #optimizer = tf.train.AdagradOptimizer(1e-3, name="optimizer").minimize(cnn.cost)
            #optimizer = tf.train.AdadeltaOptimizer(1e-1)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cnn.cost, tvars),
                                          FLAGS.max_grad_norm)
            #grads_and_vars = optimizer.compute_gradients(cnn.cost)
            optimizer.apply_gradients(zip(grads, tvars))
            train_op=optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            sess.run(tf.initialize_all_variables())

            #cur_lr = FLAGS.lr / (epoch + 1)
            #lstm.assign_new_lr(sess, cur_lr)
            epoch = 1
            LR = linear_model.LogisticRegression()
            for idx in np.arange(FLAGS.epoches):
                clf_features = []
                clf_labels = []
                for batch_data in batch_iter(zip(ori_quests, cand_quests, features, labels, ori_quests_var, cand_quests_var), FLAGS.batch_size, epoches=1):

                    ori_train, cand_train, batch_features, batch_train_labels, ori_quests_var_batch, cand_quests_var_batch = zip(*batch_data)
	            output_features = run_step(sess, ori_train, cand_train, batch_features, batch_train_labels, ori_quests_var_batch, cand_quests_var_batch, cnn, FLAGS.dropout)
                    clf_features.append(output_features)
                    clf_labels.append(batch_train_labels)
                    
                clf_features = np.concatenate(clf_features)
                clf_labels = np.concatenate(clf_labels)
            LR.fit(clf_features, clf_labels)
            #LR.fit(features, labels)

            LR_path = build_path("./models/", FLAGS.data_type, FLAGS.modelType, FLAGS.num_layers, "-" + str(epoch) + "-LR.pkl")
            joblib.dump(LR, LR_path)
            path = saver.save(sess,checkpoint_prefix,global_step)
            logger.info("finished")
            #---------------------------------- end train -----------------------------------
