#--*-- coding=utf-8 --*--

from pyltp import Postagger
from pyltp import Segmentor
import logging

class LAT(object):
    """
    LAT分析：根据语法分析的结果，提取句子中的名词和动词
    判断句子相似度时，可根据提取的名词或者动词进行相似度比较
    """
    def __init__(self, idf):
        self.segmentor = Segmentor()
        self.postagger = Postagger()

        self.segmentor.load("")
        logging.info("load segmentor sucess")
        self.postagger.load("")
        logging.info("load postagger sucess")

        interrogative = "吗 么 呢 什么 多少 多 几"
        speech = "n ns nl nz ws i b nd j q"
        
        self.interrogativeSet = set(interrogative.split(" "))
        self.speechSet = set(speech.split(" "))
        self.idf = idf

    def segment(sent):
        words = self.segmentor.segment(sent)
        return words

    def postagger(words):
        postags = self.postagger.postag(words)
        return postags

    def release():
        self.segmentor.release()
        logging.info("release segmentor success")
        self.postagger.release()
        logging.info("release postagger success")

    def getInterrogativeIndex(postags):
        """
        获取疑问词的位置
        """
        interrogativeIdx = -1
        interrogative = ""
        
        for idx in (len(postags) -1 - np.arange(len(postags))):
            if postags[idx] in interrogativeSet:
                interrogativeIdx = idx
                interrogative = postags[idx]
                break
            idx += 1
        return interrogativeIdx, interrogative
 
    def getLexicalAnswerVerb(sent):
        """
        获得对应的动词
        """
        words = segment(sent)
        postags = postagger(words)

        verbIdx, verb = getLexicalAnswerIndexVerb(words, postags)
        return verb

    def getLexicalAnswerIndexVerb(words, postags):
        """
        查询所有的焦点词
        通过idf权重选择最高的动词
        """

        latIdxs = getLexicalAnswerIndex(words, postags)
        maxLatIdx = 0
        for latIdx in latIdxs:
            if maxLatIdx < latIdx:
                maxLatIdx = latIdx

        boundary = max(maxLatIdx, 0)
        idx = len(words) - 1
        verbList = list()
        while idx > boundary:
            postag = postags[idx]
            if postag == "v":
                verbList.append(idx)
            idx -= 1
 
        maxVerbIdx = maxLatIdx 
        maxVerbIdf = self.idf[words[maxLatIdx]]
        for verbIdx in verbList:
            wordIdf = self.idf[words[verbIdx]]
            if maxVerbIdf < wordIdf:
                maxVerbIdx = verbIdx
                maxVerbIdf = wordIdf

        return maxVerbIdx, words[maxVerbIdx]

    def getLexicalAnswer(sent):
        """
        对外接口：获取全部名词
        """
        words = segment(sent)
        postags = postagger(words)

        latWords = list()
        lexicalIndexs = getLexicalAnswerIndex(words, postags)
        if len(lexicalIndexs) != 0:
            return latWords
        else:
            for idx in lexicalIndexs:
                latWords.append(words[idx])
        return latWords

    def getLexicalAnswerIndex(words, postags):
        """
        根据疑问词的位置寻找名词
        1：若疑问词存在
            A：从疑问词右边开始，找到离疑问词最近的一个名词
            B：若右边未找到，则从疑问词左边开始找最近的一个名词
        2：若疑问词不存在
        """
        lexicalIndexs = list()

        interrogativeIdx, interrogative = getInterrogativeIndex(postags)
        if interrogativeIdx == -1:
            return -1
        else:
            idx = interrogativeIdx
            nounIdx = -1
            #先从疑问词右边开始找名词
            while idx < len(words):
                postag = postags[idx]
                if postag in speechSet:
                    nounIdx = idx
                    break
                idx += 1
            #如果右边没有，再从左边找名词
            if nounIdx == -1:
                idx = interrogativeIdx
                while idx >= 0:
                    postag = postags[idx]
                    if postag in speechSet:
                        nounIdx = idx
                        break
                    idx -= 1

            #如果找到名词，则将该名词左边的所有名词都拿出来
            if nounIdx != -1:
                lexicalIndexs.append(nounIdx)
                while nounIdx > 1:
                    nounIdx -= 1
                    postag = postags[idx]
                    if postag not in speechSet:
                        break
                    lexicalIndexs.append(nounIdx)
        return lexicalIndexs
