#--*-- coding=utf-8 --*--

import jieba
import jieba.posseg as psg
import logging
import numpy as np
import codecs

class LAT(object):
    """
    LAT分析：根据语法分析的结果，提取句子中的名词和动词
    判断句子相似度时，可根据提取的名词或者动词进行相似度比较
    """
    def __init__(self):
        interrogative = u"吗 么 呢 什么 多少 多 几 啊 阿"
        speech = u"i j n nd nh ni nl ns nt nz nr q eng"
        
        self.interrogativeSet = set(interrogative.split(" "))
        self.speechSet = set(speech.split(" "))

    def segment(self, sent):
        words = jieba.cut(sent)
        return list(words)

    def postaggerMethod(self, sent):
        postags = list()
        words = list()
        for word, postag in psg.cut(sent):
            words.append(word)
            postags.append(postag)
        return words, postags

    def getLexicalAnswerVerb(self, sent, idf):
        """
        获得对应的动词
        """
        words, postags = self.postaggerMethod(sent)

        verbIdx, verb = self.getLexicalAnswerIndexVerb(words, postags, idf)
        return verb

    def getLexicalAnswerIndexVerb(self, words, postags, idf):
        """
        查询所有的焦点词
        通过idf权重选择最高的动词
        """

        latIdxs = self.getLexicalAnswerIndex(words, postags)
        maxLatIdx = 0
        for latIdx in latIdxs:
            if maxLatIdx < latIdx:
                maxLatIdx = latIdx

        word = words[maxLatIdx]
        if word not in idf or idf[word] == 0:
            return -1, ""

        boundary = max(maxLatIdx, 0)
        idx = len(words) - 1
        verbList = list()
        while idx > boundary:#查找名词右边的动词
            postag = postags[idx]
            if postag == u"v":
                verbList.append(idx)
            idx -= 1
 
        maxVerbIdx = -1
        maxVerbIdf = 0.
        for verbIdx in verbList:
            word = words[verbIdx]
            if word not in idf:
                continue
            wordIdf = idf[word]
            if maxVerbIdf < wordIdf:
                maxVerbIdx = verbIdx
                maxVerbIdf = wordIdf
        if maxVerbIdx == -1:
            return -1, ""
        else:
            return maxVerbIdx, words[maxVerbIdx]

    def getLexicalAnswer(self, sent):
        """
        对外接口：获取全部名词
        """
        words, postags = self.postaggerMethod(sent)

        latWords = list()
        lexicalIndexs = self.getLexicalAnswerIndex(words, postags)
        if len(lexicalIndexs) == 0:
            return latWords
        else:
            for idx in lexicalIndexs:
                latWords.append(words[idx])
        return latWords

    def getLexicalAnswerIndex(self, words, postags):
        """
        根据疑问词的位置寻找名词
        1：若疑问词存在
            A：从疑问词右边开始，找到离疑问词最近的一个名词
            B：若右边未找到，则从疑问词左边开始找最近的一个名词
        2：若疑问词不存在
        """
        lexicalIndexs = list()

        interrogativeIdx, interrogative = self.getInterrogativeIndex(words)
        if interrogativeIdx == -1:
            return lexicalIndexs
        else:
            idx = interrogativeIdx
            nounIdx = -1
            #先从疑问词右边开始找名词
            while idx < len(words):
                postag = postags[idx]
                if postag in self.speechSet:
                    nounIdx = idx
                    break
                idx += 1
            #如果右边没有，再从左边找名词
            if nounIdx == -1:
                idx = interrogativeIdx
                while idx >= 0:
                    postag = postags[idx]
                    if postag in self.speechSet:
                        nounIdx = idx
                        break
                    idx -= 1

            #如果找到名词，则将该名词左边的所有名词都拿出来
            if nounIdx != -1:
                lexicalIndexs.append(nounIdx)
                while nounIdx > 1:
                    nounIdx -= 1
                    postag = postags[nounIdx]
                    if postag not in self.speechSet:
                        break
                    lexicalIndexs.append(nounIdx)
        return lexicalIndexs

    def getInterrogativeIndex(self, words):
        """
        获取疑问词的位置
        """
        interrogativeIdx = -1
        interrogative = ""
        
        for idx in (len(words) -1 - np.arange(len(words))):
            if words[idx] in self.interrogativeSet:
                interrogativeIdx = idx
                interrogative = words[idx]
                break
            idx += 1
        return interrogativeIdx, interrogative

def loadTfidfFromFile(tfidfFile):
    word2weight = {}
    with codecs.open(tfidfFile, "r", "utf-8") as rf:
        for line in rf.readlines():
            arr = line.split("\n")[0].split("\t")
            word = arr[0]
            weight = float(arr[1])
            word2weight[word] = weight
    return word2weight

if __name__ == "__main__":
    lat = LAT()
    sent = "这款电脑可以挂墙上吗 ？"
    nouns = lat.getLexicalAnswer(sent)
    idf = loadTfidfFromFile("../JIMI-DATA/tfidf.txt")
    verb = lat.getLexicalAnswerVerb(sent, idf)
    print verb
    for noun in nouns:
        print noun
