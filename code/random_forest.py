#!coding=utf-8
# from progressbar import ProgressBar, Bar, Percentage
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold 

def get_classifier(is_multiclass=False):
    if is_multiclass:
        return OneVsRestClassifier(ensemble.RandomForestClassifier(n_estimators=60, max_features="auto" ,n_jobs=-1, max_depth=15))
    else:
        return ensemble.RandomForestClassifier(n_estimators=60, max_features="auto" ,n_jobs=-1, max_depth=15)

def run(featureMatrix, LabelList, testNum = 10, test_size = 0.2, is_multiclass = False):
    average = 0
    # pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=testNum).start()
    for i in range(0, testNum):  
        #加载数据集，切分数据集80%训练，20%测试  
        kf = StratifiedKFold(LabelList, round(1. /test_size))
        train_indices, valid_indices = next(iter(kf))
        x_train, y_train = featureMatrix[train_indices], LabelList[train_indices]
        x_valid, y_valid = featureMatrix[valid_indices], LabelList[valid_indices]
        #训练RF分类器  
        clf = get_classifier(is_multiclass)
        clf.fit(x_train, y_train)  
        y_pred = clf.predict(x_valid)  
        p = np.mean(y_pred == y_valid)  
        average += p
        # pbar.update(i+1)
    # pbar.finish() 
    print("average precision(random_forest):", average/testNum)

def pred(train_feature_matrix, test_feature_matrix, label_list, filename, is_multiclass = False):
    clf = get_classifier(is_multiclass)
    clf.fit(train_feature_matrix, label_list)
    test_label_pred = clf.predict(test_feature_matrix)
    save_label(test_label_pred,filename)

def save_label(label_list,filename):
	f = open(filename, 'a')
	for i in range(len(label_list)):
		f.write(str(label_list[i]))
		f.write('\n')
	f.close()
