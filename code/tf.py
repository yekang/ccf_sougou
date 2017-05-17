from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.externals import joblib as jl
from sklearn.decomposition import SparsePCA
import data_preprocess
# from numpy import *
# from scipy import * 
# from theano import sparse
def tfcount():
    gender_label_cleaned_list=[]
    age_label_cleaned_list=[]
    education_label_cleaned_list=[]
    gender_feature_cleaned_list=[]
    age_feature_cleaned_list=[]
    education_feature_cleaned_list=[]
    gender_label_list =data_preprocess.get_gender_label_list()
    age_label_list =data_preprocess.get_age_label_list()
    education_label_list = data_preprocess.get_education_label_list()
    seg_train=jl.load("docs.train")
    seg_test=jl.load("docs.test")
    for x in range(len(seg_train)):
        if gender_label_list[x] != 0:
            gender_feature_cleaned_list.append(seg_train[x])
            gender_label_cleaned_list.append(gender_label_list[x])
        if age_label_list[x] != 0:
            age_feature_cleaned_list.append(seg_train[x])
            age_label_cleaned_list.append(age_label_list[x])
        if education_label_list[x] != 0:
            education_feature_cleaned_list.append(seg_train[x])
            education_label_cleaned_list.append(education_label_list[x])
    sentence_gender_list=[]
    sentence_age_list=[]
    sentence_education_list=[]
    for i in range(len(gender_feature_cleaned_list)):
        sentence_gender_list.append(' '.join(gender_feature_cleaned_list[i]))
    for i in range(len(age_feature_cleaned_list)):
        sentence_age_list.append(' '.join(age_feature_cleaned_list[i]))
    for i in range(len(education_feature_cleaned_list)):
        sentence_education_list.append(' '.join(education_feature_cleaned_list[i]))
    ctf=CountVectorizer()
    # train_matrix = ctf.fit_transform(sentence_list)
    gender_matrix = ctf.fit_transform(sentence_gender_list)
    age_matrix = ctf.fit_transform(sentence_age_list)
    education_matrix = ctf.fit_transform(sentence_education_list)
    return gender_matrix,age_matrix,education_matrix


def tf_label():
    gender_label_list =data_preprocess.get_gender_label_list()
    age_label_list =data_preprocess.get_age_label_list()
    education_label_list = data_preprocess.get_education_label_list()
    gender_label_cleaned_list=[]
    age_label_cleaned_list=[]
    education_label_cleaned_list=[]
    seg_train=jl.load("docs.train")
    seg_test=jl.load("docs.test")
    for x in range(len(seg_train)):
        if gender_label_list[x] != 0:
            # gender_feature_cleaned_list.append(seg_train[x])
            gender_label_cleaned_list.append(gender_label_list[x])
        if age_label_list[x] != 0:
            # age_feature_cleaned_list.append(seg_train[x])
            age_label_cleaned_list.append(age_label_list[x])
        if education_label_list[x] != 0:
            # education_feature_cleaned_list.append(seg_train[x])
            education_label_cleaned_list.append(education_label_list[x])
    return gender_label_cleaned_list,age_label_cleaned_list,education_label_cleaned_list

# if __name__ == '__main__':
    # train_matrix=main()
    # jl.dump(train_matrix,'tf_train')
    # train_matrix_tem=jl.load('tf_train')
    # train_matrix=train_matrix_tem.transpose()
    # pca_mtrix = toPCA()
    # print(pca_matrix)

    