import gensim,logging
import data_preprocess
def sentence():
    querydoc_list=data_preprocess.load_querydoc('fenciwordstest_noeng.txt')
    return querydoc_list
def word_vec(setence_list):
    model_train4test=gensim.models.Word2Vec(setence_list,min_count=1)
    model_train4test.save('./word2vec/test_vec')
    print(model_train4test['英雄'])
    return model_train4test
# def test(model):
    

if __name__ == '__main__':
    querydoc_list=sentence()
    word_vec(querydoc_list)