#!coding=utf-8
import gensim
from gensim import corpora, models, similarities
# from progressbar import ProgressBar,Bar,Percentage
import numpy as np

def load_corpus():
    return corpora.MmCorpus('topic_model/corpus.mm')

def load_model():
    return models.LdaModel.load('topic_model/model.lda')

def print_topics(model):
    for topic_id, words in model.print_topics(100,20):
        print(str(topic_id) + ": ")
        print(str(words))

def build_model(dataset, num_topics=100, is_hdp=True):
    print("generating dictionary and corpus...")
    dic = corpora.Dictionary(dataset)
    dic.filter_extremes(no_below=2) # 去除低频词汇
    corpus = [dic.doc2bow(text) for text in dataset]
    print("constructing LDA model...")
    if is_hdp:
        hdp = models.HdpModel(corpus, id2word=dic)
        (alpha, beta) = hdp.hdp_to_lda()
        model = models.LdaModel(id2word=hdp.id2word, num_topics=len(alpha),
                                alpha=alpha, eta=hdp.m_eta)
        model.expElogbeta = np.array(beta, dtype=np.float32)
        num_topics = len(alpha)
    else:
        model = models.LdaMulticore(corpus, id2word=dic, num_topics=num_topics)
    print("saving model...")
    dic.save_as_text("topic_model/dic.txt")
    corpora.MmCorpus.serialize("topic_model/corpus.mm", corpus)
    model.save('topic_model/model.lda')

    return model, num_topics

def svec2dense(svec, maxval):
    dense = [0] * maxval;
    for elem in svec:
        dense[elem[0]] = elem[1]
    return dense

def extract_sprase_feature_matrix(model, corpus, num_topics):
    """ """
    total_feature_matrix = []
    n = len(corpus)
    #num_topics = len(model.alpha)
    # pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=n).start()
    for i in range(n):
        total_feature_matrix.append(svec2dense(model.get_document_topics(corpus[i], minimum_probability = 0.01), num_topics))
        # pbar.update(i+1)
    # pbar.finish()
    save_feature_matrix(total_feature_matrix,"feature_matrix/total_feature_matrix.txt")
    save_feature_matrix(total_feature_matrix[:20000],"feature_matrix/train_feature_matrix.txt")
    save_feature_matrix(total_feature_matrix[20000:],"feature_matrix/test_feature_matrix.txt")

def extract_feature_matrix(model, corpus):
    total_feature_matrix = []
    n = len(corpus)
    # pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=n).start()
    for i in range(n):
        total_feature_matrix.append([p*100 for _,p in model.get_document_topics(corpus[i], minimum_probability = 0.0)])
        # pbar.update(i+1)
    # pbar.finish()
    save_feature_matrix(total_feature_matrix,"feature_matrix/total_feature_matrix.txt")
    save_feature_matrix(total_feature_matrix[:20000],"feature_matrix/train_feature_matrix.txt")
    save_feature_matrix(total_feature_matrix[20000:],"feature_matrix/test_feature_matrix.txt")



def save_feature_matrix(feature_matrix,filename):
    n = len(feature_matrix)
    m = len(feature_matrix[0])
    f = open(filename, 'a')
    for i in range(n):
        for j in range(m):
            f.write(str(feature_matrix[i][j]) + ' ')
        f.write('\n')
    f.close()

def get_train_feature_matrix():
    return load_feature_matrix("feature_matrix/train_feature_matrix.txt")

def get_test_feature_matrix():
    return load_feature_matrix("feature_matrix/test_feature_matrix.txt")

def get_total_feature_matrix():
    return load_feature_matrix("feature_matrix/total_feature_matrix.txt")

def load_feature_matrix(filename):
	feature_matrix = []
	feature_temp_list = open(filename, 'r').readlines()
	feature_matrix = [[float(feature) for feature in feature_temp.split()] for feature_temp in feature_temp_list]
	return feature_matrix

def clean(featureMatrix,labelList):
    new_featureMatrix = []
    new_labelList = []
    for i in range(len(labelList)):
        if int(labelList[i]) != 0:
            new_featureMatrix.append(featureMatrix[i])
            new_labelList.append(int(labelList[i]))
    return np.array(new_featureMatrix), np.array(new_labelList)