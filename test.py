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

from data_helper import load_train_data, load_test_data, load_embedding, batch_iter, cal_basic_feature, gen_neg_quest
from polymerization import CNN_QA
from utils import build_path
from preprocess import Word2Vec, MSRP, WikiQA


#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../insuranceQA/train", "train corpus file")
tf.flags.DEFINE_string("test_file", "../insuranceQA/test1", "test corpus file")
tf.flags.DEFINE_string("valid_file", "../insuranceQA/test1.sample", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "../insuranceQA/vectors.nobin", "embedding file")
tf.flags.DEFINE_integer("embedding_size", 300, "embedding size")
tf.flags.DEFINE_integer("num_filters", 100, "embedding size")
tf.flags.DEFINE_string("filter_size", "3", "embedding size")
tf.flags.DEFINE_float("dropout", 0.5, "the proportion of dropout")
tf.flags.DEFINE_float("lr", 0.1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 64, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 50, "epoches")
tf.flags.DEFINE_integer("evaluate_every", 1000, "run evaluation")
tf.flags.DEFINE_integer("seq_len", 30, "embedding size")
tf.flags.DEFINE_integer("num_layers", 2, "embedding size")
tf.flags.DEFINE_string("out_dir", "save/", "output directory")
tf.flags.DEFINE_string("modelType", "BCNN", "output directory")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.2, "use memory rate")

FLAGS = tf.flags.FLAGS
filter_size = int(FLAGS.filter_size)
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger -------------------------------
logger = logging.getLogger("test")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./test.log", mode="w")
fh.setLevel(logging.INFO)

logger.addHandler(fh)
#----------------------------- define a logger end ----------------------------------

#------------------------------------load data -------------------------------
#load data
word2Vec = Word2Vec()
data_type = "WikiQA"
if data_type == "WikiQA":
    test_data = WikiQA(word2vec=word2Vec)
else:
    test_data = MSRP(word2vec=word2Vec)
test_data.open_file(mode="test")
#----------------------------------- load data end ----------------------

#----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:1"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cnn = CNN_QA(FLAGS.batch_size, test_data.max_len, FLAGS.embedding_size, filter_size, FLAGS.num_filters, test_data.num_features, FLAGS.num_layers, model_type=FLAGS.modelType)
            global_step = tf.Variable(0, name="globle_step",trainable=False)
            #add checkpoint
            saver = tf.train.Saver(tf.all_variables())
            checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            LR_path = build_path("./models/", data_type, FLAGS.modelType, FLAGS.num_layers, "-" + str(FLAGS.epoches) + "-LR.pkl")
            lr = joblib.load(LR_path)

            QA_pairs = {}
            s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size)

            for i in range(test_data.data_size):
                pred, clf_input = sess.run([cnn.prediction, cnn.output_features],
                                           feed_dict={cnn.ori_input_quests: np.expand_dims(s1s[i], axis=0),
                                                      cnn.cand_input_quests: np.expand_dims(s2s[i], axis=0),
                                                      cnn.labels: np.expand_dims(labels[i], axis=0),
                                                      cnn.features: np.expand_dims(features[i], axis=0)})

                clf_pred = lr.predict_proba(clf_input)[:, 1]
                pred = clf_pred

                s1 = " ".join(test_data.s1s[i])
                s2 = " ".join(test_data.s2s[i])

                if s1 in QA_pairs:
                    QA_pairs[s1].append((s2, labels[i], np.asscalar(pred)))
                else:
                    QA_pairs[s1] = [(s2, labels[i], np.asscalar(pred))]

            # Calculate MAP and MRR for comparing performance
            MAP, MRR = 0, 0
            for s1 in QA_pairs.keys():
                p, AP = 0, 0
                MRR_check = False

                QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)

                for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                    if label == 1:
                        if not MRR_check:
                            MRR += 1 / (idx + 1)
                            MRR_check = True

                        p += 1
                        AP += p / (idx + 1)

                AP /= p
                MAP += AP

            num_questions = len(QA_pairs.keys())
            MAP = float(MAP) / num_questions
            MRR = float(MRR) / num_questions

            logger.info("[MAP:" + str(MAP) +  "/ MRR:" + str(MRR))

            logger.info("finished")
            #---------------------------------- end train -----------------------------------
