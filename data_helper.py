#coding=utf-8

import codecs
import logging
import numpy as np
import os
import jieba
import jieba.analyse 
import traceback
import math
from LAT import LAT

from collections import defaultdict
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

from collections import defaultdict
from utils import np_cos_sim, lcs_continuous, lcs


# define a logger
logging.basicConfig(format="%(message)s", level=logging.INFO)

def init_feature(num):
    features = []
    for idx in np.arange(num):
        features.append([])
    return features

def cal_Lcs(ori_quests, cand_quests, features):
    """
    calculate lcs of original question and candicate question
    """
    for idx in np.arange(len(ori_quests)):
        ori_quest = [each for each in ori_quests[idx]]
        cand_quest = [each for each in cand_quests[idx]]
        score = lcs_continuous(ori_quest, cand_quest) * 2. / (len(ori_quest) + len(cand_quest))
        features[idx].append(score)

def cal_length(ori_quests, features, max_len = 50):
    """
    calculate the length of original question and candicate question
    """
    for idx in np.arange(len(ori_quests)):
        ori_quest = [each for each in ori_quests[idx]]
        score = float(len(ori_quest)) / max_len
        features[idx].append(score)

def cal_word2vec_sim(ori_quest, cand_quest, embeddings, word2idx):
    """
    calculate word2vec similarity of original question and candicate question
    """
    unknownId = word2idx.get("UNKNOWN", 0)
    ori_quest_emb = [embeddings[word2idx.get(word, unknownId)] for word in ori_quest if len(word) != 0]
    cand_quest_emb = [embeddings[word2idx.get(word, unknownId)] for word in cand_quest if len(word) != 0]
    if len(ori_quest_emb) != 0 and len(cand_quest_emb) != 0:
        ori_mean_emb = np.mean(ori_quest_emb, 0)
        cand_mean_emb = np.mean(cand_quest_emb, 0)
        score = np_cos_sim(ori_mean_emb, cand_mean_emb)
    else:
        score = 0
    return score

def cal_tf(word, count):
    return float(count.get(word)) / sum(count.values())

def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)

def cal_idf(word, count_list):
    return math.log(float(len(count_list)) / (1 + n_containing(word, count_list)))

def cal_tfidf(word, count, count_list):
    return cal_tf(word, count) * cal_idf(word, count_list)

def cal_TFIDF(total_quests):
    """
    calculate tf-idf 
    """
    corpus = list()
    words = set()
    for sents in total_quests:
        tf = defaultdict()
        arr = sents.split(" ")
        for word in arr:
            if len(word.strip()) != 0:
                words.add(word)
                tf.setdefault(word, 0)

                tf[word] += 1
        corpus.append(tf)

    word_idfs = defaultdict()
    for word in words:
        word_idf = cal_idf(word, corpus)
        word_idfs[word] = word_idf
    
    return corpus, word_idfs

def cal_TFIDF_sim(ori_quest, cand_quest, corpus):
    """
    calculate tf-idf similarity between original question and candicate question
    """
    try:
        molecular, denominator = 0, 0
        ori_quest_norm, cand_quest_norm = 0, 0
        ori_dict, cand_dict = defaultdict(), defaultdict()
        for idx in np.arange(len(ori_quest)):
            ori_dict.setdefault(ori_quest[idx], 0)
            ori_dict[ori_quest[idx]] += 1

        for idx in np.arange(len(cand_quest)):
            cand_dict.setdefault(cand_quest[idx], 0)
            cand_dict[cand_quest[idx]] += 1

        for idx in np.arange(len(ori_quest)):
            if ori_quest[idx] in cand_quest:
                if len(ori_quest[idx]) != 0:
                    molecular += np.power(cal_tfidf(ori_quest[idx], ori_dict, corpus), 2)
            ori_quest_norm += np.power(cal_tfidf(ori_quest[idx], ori_dict, corpus), 2)

        for idx in np.arange(len(cand_quest)):
            cand_quest_norm += np.power(cal_tfidf(cand_quest[idx], cand_dict, corpus), 2)

        denominator = np.sqrt(ori_quest_norm) * np.sqrt(cand_quest_norm)
        score = float(molecular) / denominator
        return score
    except Exception, e:
        logging.error("cal_sent_word2vec_sim error" + traceback.format_exc())
        return 0
  
    

def cal_core_term_TFIDF_sim(ori_quests, cand_quests, corpus, features, topK = 2):
    """
    calculate core term by TF-IDF
    """
    for idx in np.arange(len(ori_quests)):
        ori_tags = jieba.analyse.extract_tags(ori_quests[idx], topK=topK)
        cand_tags = jieba.analyse.extract_tags(cand_quests[idx], topK=topK)
        score = cal_TFIDF_sim(ori_tags, cand_tags, corpus)
        features[idx].append(score)

def cal_sent_TFIDF_sim(ori_quests, cand_quests, word2weight, features):
    """
    calculate core term by TF-IDF
    """
    for idx in np.arange(len(ori_quests)):
        score = cal_TFIDF_sim(ori_quests[idx], cand_quests[idx], word2weight)
        features[idx].append(score)

def cal_core_term_word2vec_sim(ori_quests, cand_quests, embeddings, word2idx, features, topK = 2):
    """
    calculate core term by TF-IDF
    """
    for idx in np.arange(len(ori_quests)):
        ori_tags = jieba.analyse.extract_tags(ori_quests[idx], topK=topK)
        cand_tags = jieba.analyse.extract_tags(cand_quests[idx], topK=topK)
        score = cal_word2vec_sim(cand_tags, ori_tags, embeddings, word2idx)
        features[idx].append(score)

def cal_sent_word2vec_sim(ori_quests, cand_quests, embeddings, word2idx, features):
    """
    calculate core term by TF-IDF
    """
    if len(ori_quests) != len(cand_quests):
        logging.info("original input not equal candicate input")
    else:
        logging.info("original input equal candicate input")
        try:
            for idx in np.arange(len(ori_quests)):
                score = cal_word2vec_sim(ori_quests[idx], cand_quests[idx], embeddings, word2idx)
                features[idx].append(score)
        except Exception, e:
            logging.error("cal_sent_word2vec_sim error" + traceback.format_exc())

def cal_LAT_noun(ori_quests, cand_quests, lat, embeddings, word2idx, features):
    """
    calculate LAT noun term similarty 
    """
    try:
        for idx in np.arange(len(ori_quests)):
            ori_quest = ori_quests[idx]
            cand_quest = cand_quests[idx]
            ori_noun = lat.getLexicalAnswer(ori_quest)
            cand_noun = lat.getLexicalAnswer(cand_quest)
            score = cal_word2vec_sim(ori_noun, cand_noun, embeddings, word2idx)
            features[idx].append(score)
    except Exception, e:
        logging.error("cal_lat_noun_sim error" + traceback.format_exc())
    

def cal_LAT_verb(ori_quests, cand_quests, lat, embeddings, word2idx, features):
    """
    calculate LAT verb term similarty 
    """
    try:
        for idx in np.arange(len(ori_quests)):
            ori_quest = ori_quests[idx]
            cand_quest = cand_quests[idx]
            ori_verb = lat.getLexicalAnswerVerb(ori_quest)
            cand_verb = lat.getLexicalAnswerVerb(cand_quest)
            score = cal_word2vec_sim(ori_verb, cand_verb, embeddings, word2idx)
            features[idx].append(score)
    except Exception, e:
        logging.error("cal_lat_verb_sim error" + traceback.format_exc())

def cal_same_terms(ori_quests, cand_quests, features):
    """
    the number of same terms between original question and candicate question
    """
    for idx in np.arange(len(ori_quests)):
        ori_quest = [each for each in ori_quests[idx]]
        cand_quest = [each for each in cand_quests[idx]]
        _, commList = lcs(ori_quest, cand_quest)
        score = float(len(commList)) / len(ori_quest)
        features[idx].append(score)

def cal_max_similarity_term(ori_quests, cand_quests, features):
    """
    the length of the same terms between original question and candicate question
    """
    for idx in np.arange(len(ori_quests)):
        ori_quest = [each for each in ori_quests[idx]]
        cand_quest = [each for each in cand_quests[idx]]
        _, commList = lcs(ori_quest, cand_quest)
        commLength, oriLen = 0, 0
        for each in commList:
            commLength += len(each)
        for each in ori_quest:
            oriLen += len(each)
        score = float(commLength) / oriLen
        features[idx].append(score)

def load_embedding(filename, embedding_size):
    """
    load embedding
    """
    embeddings = []
    word2idx = defaultdict(list)
    idx2word = defaultdict(list)
    idx = 0
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                idx += 1
                arr = line.split(" ")
                if len(arr) != (embedding_size + 1):
                    logging.error("embedding error, index is:%s"%(idx))
                    continue

                embedding = [float(val) for val in arr[1 : ]]
                word2idx[arr[0]] = len(word2idx)
                idx2word[len(word2idx)] = arr[0]
                embeddings.append(embedding)

        except Exception, e:
            logging.error("cal basic features error" + traceback.format_exc())
        finally:
            rf.close()

    word2idx["UNKNOWN"] = len(idx2word)
    idx2word[len(idx2word)] = "UNKNOWN"
    word2idx["<a>"] = len(idx2word)
    idx2word[len(idx2word)] = "<a>"

    unknown_padding_embedding = np.random.normal(0, 0.1, (2,embedding_size))
    embs = np.append(embeddings, unknown_padding_embedding.astype(np.float32), axis=0)

    logging.info("load embedding finish!")
    return embs, word2idx, idx2word

def sent_to_idx(sent, word2idx, sequence_len):
    """
    convert sentence to index array
    """
    unknown_id = word2idx.get("UNKNOWN", 0)
    pad_id = word2idx.get("<a>", 0)
    sent2idx = [word2idx.get(word, unknown_id) for word in sent.split(" ")]
    if len(sent2idx) < sequence_len:
        sent2idx_pad = np.concatenate([sent2idx, [pad_id] * (sequence_len - len(sent2idx))])
    else:
        sent2idx_pad = sent2idx[:sequence_len]
    return sent2idx, sent2idx_pad

def load_train_data(filename, word2idx, quest_len, answer_len):
    """
    load train data
    """
    ori_quests, cand_quests, ori_quests_var, cand_quests_var, labels, total_quests = [], [], [], [], [], []
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                arr = line.strip().split("\n")[0].split("\t")
                if len(arr) != 3:# or arr[0] != "1":
                    logging.error("invalid data:%s"%(line))
                    continue

                ori_quest, ori_quest_pad = sent_to_idx(arr[0], word2idx, quest_len)
                cand_quest, cand_quest_pad = sent_to_idx(arr[1], word2idx, answer_len)
                label = int(arr[2])
                if label not in[0, 1]:
                    continue

                ori_quests.append(ori_quest_pad)
                cand_quests.append(cand_quest_pad)
                ori_quests_var.append(arr[0])
                cand_quests_var.append(arr[1])
                total_quests.append("".join(arr[0].split(" ")))
                total_quests.append("".join(arr[1].split(" ")))
                labels.append(label)

        except Exception, e:
            logging.error("cal basic features error" + traceback.format_exc())
        finally:
            rf.close()
    logging.info("load train data finish!")

    return ori_quests, cand_quests, labels, ori_quests_var, cand_quests_var, total_quests

def create_valid(data, proportion=0.1):
    if data is None:
        logging.error("data is none")
        os._exit(1)

    data_len = len(data)
    shuffle_idx = np.random.permutation(np.arange(data_len))
    data = np.array(data)[shuffle_idx]
    seperate_idx = int(data_len * (1 - proportion))
    return data[:seperate_idx], data[seperate_idx:]

def load_test_data(filename, word2idx, quest_len, answer_len):
    """
    load test data
    """
    ori_quests, cand_quests, labels, ori_quests_var, cand_quests_var, total_quests = [], [], [], [], [], []
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                arr = line.strip().split("\n")[0].split("\t")
                if len(arr) != 4:
                    logging.error("invalid data:%s"%(line))
                    continue

                ori_quest, ori_quest_pad = sent_to_idx(arr[0], word2idx, quest_len)
                cand_quest, cand_quest_pad = sent_to_idx(arr[1], word2idx, answer_len)
                label = int(arr[3])

                ori_quests.append(ori_quest_pad)
                cand_quests.append(cand_quest_pad)
                ori_quests_var.append(arr[0])
                cand_quests_var.append(arr[1])
                total_quests.append("".join(arr[0].split(" ")))
                total_quests.append("".join(arr[1].split(" ")))
                labels.append(label)
        except Exception, e:
            logging.error("cal basic features error" + traceback.format_exc())
        finally:
            rf.close()
    logging.info("load test data finish!")
    return ori_quests, cand_quests, labels, ori_quests_var, cand_quests_var, total_quests

def gen_neg_quest(ori_quests, cand_quests, neg_num = 5):
    num = len(ori_quests)
    corpus = list()
    if num != 0:
        for idx in np.arange(num):
            corpus.append((ori_quests[idx], cand_quests[idx], 1))
            randi_list = list()
            [randi_list.append(r_idx) for r_idx in np.random.randint(0, num, 5 * neg_num) if r_idx != idx and len(randi_list) < neg_num]
            for each in randi_list:
                corpus.append((ori_quests[idx], cand_quests[each], 0))

    return parseTuple(corpus) 

def parseTuple(corpus):
    ori_quests, cand_quests, labels = list(), list(), list()
    for ori, cand, label in corpus:
        ori_quests.append(ori)
        cand_quests.append(cand)
        labels.append(label)
    return ori_quests, cand_quests, labels

def saveTFIDF2File(word2weights, filename):
    with codecs.open(filename, "w", "utf-8") as wf:
        for word2weight in word2weights.items():
            word = word2weight[0]
            weight = word2weight[1]
            wf.write(word + "\t" + str(weight) + "\n")

def loadTfidfFromFile(tfidfFile):
    word2weight = {}
    with codecs.open(tfidfFile, "r", "utf-8") as rf:
        for line in rf.readlines():
            arr = line.split("\n")[0].split("\t")
            word = arr[0]
            weight = float(arr[1])
            word2weight[word] = weight
    return word2weight

def cal_basic_feature(ori_quests, cand_quests, total_quests, embeddings, word2idx, saveTfidf = False, tfidfFile = None):
    """
    calculate basic featrue
    1:the length of original question
    2:the length of candicate question
    3:the lcs between original question and candicate question
    4:the tf-idf between original question and candicate question
    5:the word2vec similarity between original question and candicate question
    6:the word2vec similarity between the LAT of original question and the LAT of candicate question
    7:the word2vec similarity between the LAT term of original question and the LAT term of candicate question
    8:the word2vec similarity between the core term of original question and the core term of candicate question
    9:the tf-idf similarity between the core term of original question and the core term of candicate question
    10:the length of same terms between original question and candicate question
    11:the length of the max-similarity terms between original question and candicate question
    """
    features = list()
    num = len(ori_quests)
    try:
        if num != 0:
            total_sents = list()
            ori_sents, cand_sents = segment(ori_quests, cand_quests)
            #ori_sents, cand_sents = ori_quests, cand_quests
            total_sents.extend(ori_quests)
            total_sents.extend(cand_quests)
            if saveTfidf:
                corpus, word_idfs = cal_TFIDF(total_sents)
                saveTFIDF2File(word_idfs, tfidfFile)
            else:
                word_idfs = loadTfidfFromFile(tfidfFile)
            logging.info("init idf success")
            features = init_feature(len(ori_sents))

            cal_Lcs(ori_sents, cand_sents, features)
            logging.info("calculate lcs feature success")
            cal_length(ori_sents, features)
            logging.info("calculate length of original question feature success")
            cal_length(cand_sents, features)
            logging.info("calculate length of candicate question feature success")
            cal_same_terms(ori_sents, cand_sents, features)
            logging.info("calculate same term feature success")
            cal_max_similarity_term(ori_sents, cand_sents, features)
            logging.info("calculate max similarity of term feature success")
            #cal_core_term_TFIDF_sim(ori_quests, cand_quests, corpus, features, topK = 2)
            cal_core_term_word2vec_sim(ori_quests, cand_quests, embeddings, word2idx, features, topK = 2)
            logging.info("calculate core term word2vec feature success")
            cal_sent_TFIDF_sim(ori_sents, cand_sents, word_idfs, features)
            logging.info("calculate sentence tfidf feature success")
            cal_sent_word2vec_sim(ori_sents, cand_sents, embeddings, word2idx, features)
            logging.info("calculate sentence word2vec similarity feature success")
            lat = LAT(word_idfs)
            logging.info("feature size:" + str(features[0]))
        else:
            logging.error("cal basic features error" + traceback.format_exc())
    except Exception, e:
        logging.error("cal basic features error" + traceback.format_exc())
    return features

def segment(ori_quests, cand_quests):
    #jieba.enable_parallel(20)
    ori_terms, cand_terms = list(), list()
    for idx in np.arange(len(ori_quests)):
        #ori_quest = [each for each in jieba.cut(ori_quests[idx])]
        #cand_quest = [each for each in jieba.cut(cand_quests[idx])]
        ori_quest = ori_quests[idx].split(" ")
        cand_quest = cand_quests[idx].split(" ")
        
        ori_terms.append(ori_quest)
        cand_terms.append(cand_quest)
    return ori_terms, cand_terms


def batch_iter(data, batch_size, epoches, shuffle=True):
    """
    iterate the data
    """
    data_len = len(data)
    batch_num = int(data_len / batch_size) 
    data = np.array(data)

    for epoch in range(epoches):
        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(data_len))
            shuffle_data = data[shuffle_idx]
        else:
            shuffle_data = data

        for batch in range(batch_num):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, data_len)
            yield shuffle_data[start_idx : end_idx]
