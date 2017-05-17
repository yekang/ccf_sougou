#!coding=utf-8
# from progressbar import ProgressBar, Bar, Percentage
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def run(featureMatrix, LabelList, testNum = 10, test_size = 0.2):
    average = 0
    # pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=testNum).start()
    for i in range(0, testNum):  
        #加载数据集，切分数据集80%训练，20%测试  
        kf = StratifiedKFold(LabelList, round(1. /test_size))
        train_indices, valid_indices = next(iter(kf))
        x_train, y_train = featureMatrix[train_indices], LabelList[train_indices]
        x_valid, y_valid = featureMatrix[valid_indices], LabelList[valid_indices]
        #训练LR分类器  
        clf = KNeighborsClassifier()  
        clf.fit(x_train, y_train)  
        y_pred = clf.predict(x_test)  
        p = np.mean(y_pred == y_test)  
        average += p
        # pbar.update(i+1)
    # pbar.finish() 
    print("average precision(knn):", average/testNum)
    return average/testNum

def pred(train_feature_matrix, test_feature_matrix, label_list, filename):
    clf = LogisticRegression()
    clf.fit(train_feature_matrix, label_list)
    test_label_pred = clf.predict(test_feature_matrix)
    save_label(test_label_pred,filename)

def save_label(label_list,filename):
	f = open(filename, 'a')
	for i in range(len(label_list)):
		f.write(str(label_list[i]))
		f.write('\n')
	f.close()