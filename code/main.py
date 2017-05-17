#!coding=utf-8
import data_preprocess
import topic_model
import classifier
# from progressbar import ProgressBar, Bar, Percentage
import numpy as np
import sys
import time
from sklearn.externals import joblib as jl

def log(num_topics, gender_accuracy, age_accuracy, education_accuracy):
    f = open("log.txt", 'a')
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    f.write(str(current_time) + "\n")
    f.write("model:topic_" + str(num_topics) + str(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))))
    f.write("feature_matrix:feature_matrix_100" + str(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))))
    f.write("gender:" + str(gender_accuracy))
    f.write("age:" + str(age_accuracy))
    f.write("education:" + str(education_accuracy))
    f.write("average:" + str((gender_accuracy + age_accuracy + education_accuracy) / 3))
    f.close()


def main():
    # step1: 数据预处理并保存 

    # data_preprocess.load_dataset()

    # step2：训练主题模型并保存
    # total_querydoc = data_preprocess.get_total_querydoc()
    # model, num_topics = topic_model.build_model(total_querydoc)
    # model = topic_model.load_model()
    # topic_model.print_topics(model)

    # step3: 提取分类所需特征矩阵并保存
    # model = topic_model.load_model()
    # corpus = topic_model.load_corpus()
    # topic_model.extract_sprase_feature_matrix(model, corpus, num_topics)
    # topic_model.extract_feature_matrix(model,corpus)
    # tf_train_matrix = tf.tfcount()
    tf_idf_gender_matrix,tf_idf_age_matrix,tf_idf_education_matrix = classifier.tfidf.tfidf_feature()
    gender_label_list,age_label_list,education_label_list =classifier.tfidf.tfidf_label()
    # tf_gender_matrix,tf_age_matrix,tf_education_matrix = classifier.tfidf.tfidf_feature()
    
    tf_idf_gender_matrix_cleaned=tf_idf_gender_matrix[0:len(gender_label_list)]
    tf_idf_age_matrix_cleaned=tf_idf_age_matrix[0:len(age_label_list)]
    tf_idf_education_matrix_cleaned=tf_idf_education_matrix[0:len(education_label_list)]



    # step4: 分类
    # gender_train_list=jl.load('gender_train.jl')
    # age_train_list=jl.load('age_train.jl')
    # edu_train_list=jl.load('edu_train.jl')
    # edu_label_list=(np.array(edu_train_list)[:,-1])
    # age_label_list=(np.array(age_train_list)[:,-1])
    # gender_label_list=(np.array(gender_train_list)[:,-1])
    # train_gender_efeature_matrix = np.array(gender_train_list)[:,:-1]
    # train_age_feature_matrix = np.array(age_train_list)[:,:-1]
    # train_education_feature_matrix = np.array(edu_train_list)[:,:-1]
    # gender_label_list =np.array(data_preprocess.get_gender_label_list())
    # age_label_list =np.array(data_preprocess.get_age_label_list())
    # education_label_list = np.array(data_preprocess.get_education_label_list())
    gender_label_list,age_label_list,education_label_list =classifier.tfidf.tfidf_label()
    # train_feature_matrix = topic_model.get_train_feature_matrix()
    print('predict gender :')
    # train_gender_efeature_matrix, gender_label_list = topic_model.clean(tf_idf_train_matrix,gender_label_list)
    # logistic_gender_accuracy = classifier.logistic.run(tf_idf_gender_matrix_cleaned, np.array(gender_label_list),testNum=10)
    # random_gender_accuracy = classifier.random_forest.run(tf_idf_train_matrix, gender_label_list,testNum=10)
    svc_gender_accuracy = classifier.svc.run(tf_idf_gender_matrix_cleaned, np.array(gender_label_list),testNum=10)
    # nb_gender_accuracy = classifier.naive_bayes.run(tf_idf_gender_matrix_cleaned, np.array(gender_label_list),testNum=10)
    print('predict age :')
    # train_age_feature_matrix, age_label_list = topic_model.clean(tf_idf_train_matrix,age_label_list)
    # logistic_age_accuracy = classifier.logistic.run(tf_idf_train_matrix, age_label_list,testNum=10)
    # random_age_accuracy = classifier.random_forest.run(tf_idf_train_matrix, age_label_list,testNum=10, is_multiclass = True)
    svc_age_accuracy = classifier.svc.run(tf_idf_age_matrix_cleaned, np.array(age_label_list),testNum=10)
    # nb_age_accuracy = classifier.naive_bayes.run(tf_idf_age_matrix_cleaned, np.array(age_label_list),testNum=10)
    print('predict education :')
    # train_education_feature_matrix, education_label_list = topic_model.clean(tf_idf_train_matrix,education_label_list)
    # logistic_age_accuracy = classifier.logistic.run(tf_idf_train_matrix, education_label_list,testNum=10)
    # random_education_accuracy = classifier.random_forest.run(tf_idf_train_matrix, education_label_list, testNum=10,is_multiclass = True)
    svc_education_accuracy = classifier.svc.run(tf_idf_education_matrix_cleaned,  np.array(education_label_list),testNum=10)
    # nb_education_accuracy = classifier.naive_bayes.run(tf_idf_education_matrix_cleaned, np.array(education_label_list),testNum=10)
    
    # log
    # log(20, gender_accuracy, age_accuracy, education_accuracy)

def test():
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    print(current_time)

if __name__ == '__main__':
    startTime = time.clock()
    # reload(sys)
    # sys.setdefaultencoding("utf8")
    main()
    # test()
    endTime = time.clock()
    print('Running time: %f Seconds' % (endTime-startTime))