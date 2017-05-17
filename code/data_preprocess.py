#!coding=utf-8
from numpy import *
import codecs
import jieba
import re
# from progressbar import ProgressBar,Bar,Percentage

def load_dataset():
	train_user_id_list = []
	test_user_id_list = []
	age_label_list = []
	gender_abel_list = []
	eduction_label_list = []
	train_querydoc = []
	test_querydoc = []

	# 读取源数据
	train_data_temp_list = codecs.open("dataset/user_tag_query.2W.TRAIN").readlines()
	test_data_temp_list = codecs.open("dataset/user_tag_query.2W.TEST").readlines()
	num_train = len(train_data_temp_list)
	num_test = len(test_data_temp_list)

	# 读取停用词
	stopwordsList = [str(word.decode('utf-8').strip()) for word in open('stopwords.txt').readlines()]
	stopwords_set = set(stopwordsList)

	# 处理train数据
	print('处理train数据:')
	# pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=num_train).start()
	for i in range(num_train):
		try:
			splited_temp_list = train_data_temp_list[i].decode('gb18030').split()
			train_user_id_list.append(splited_temp_list[0])
			age_label_list.append(splited_temp_list[1])
			gender_abel_list.append(splited_temp_list[2])
			eduction_label_list.append(splited_temp_list[3])
			querydoc = []
			for query in splited_temp_list[4:]:
				word_list = [str(word) for word in jieba.cut(query) if str(word) not in stopwords_set and len(word)>1] # 调用segment函数，将一条query分词，去停用词，去长度为1的词
				querydoc.extend(word_list)
			train_querydoc.append(querydoc)
		except:
			print('处理train数据失败: %d' % i)
		# finally:
			# pbar.update(i+1)
	# pbar.finish()

	# 处理test数据
	print('处理test数据:')
	# pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=num_test).start()
	for i in range(num_test):
		try:
			splited_temp_list = test_data_temp_list[i].decode('gb18030').split()
			test_user_id_list.append(splited_temp_list[0])
			querydoc = []
			for query in splited_temp_list[1:]:
				word_list = [str(word) for word in jieba.cut(query) if str(word) not in stopwords_set and len(word)>1] # 调用segment函数，将一条query分词，去停用词，去长度为1的词
				querydoc.extend(word_list)
			test_querydoc.append(querydoc)
		except:
			print('处理test数据失败: %d' % i)
		# finally:
			# pbar.update(i+1)
	# pbar.finish()

	# 缓存数据
	save_label(train_user_id_list,"dataset/train_user_id_list.txt")
	save_label(test_user_id_list,"dataset/test_user_id_list.txt")
	save_label(age_label_list,"dataset/age_label_list.txt")
	save_label(gender_abel_list,"dataset/gender_abel_list.txt")
	save_label(eduction_label_list,"dataset/eduction_label_list.txt")
	save_querydoc(train_querydoc,"dataset/train_querydoc.txt")
	save_querydoc(test_querydoc,"dataset/test_querydoc.txt")

	# 返回数据
	return train_user_id_list, test_user_id_list, age_label_list, gender_abel_list, eduction_label_list, train_querydoc, test_querydoc


def segment(sentence):
	# 读取停用词
	stopwordsList = [str(word.decode('utf-8').strip()) for word in open('stopwords.txt').readlines()]
	stopwords_set = set(stopwordsList)
	word_list = jieba.cut(sentence)
	return [str(word) for word in word_list if str(word) not in stopwords_set and len(str(word.decode('utf8')))>1]

def save_label(label_list,filename):
	f = open(filename, 'a')
	for i in range(len(label_list)):
		f.write(str(label_list[i]))
		f.write('\n')
	f.close()

def save_querydoc(querydoc_list,filename):
	f = open(filename, 'a')
	for i in range(len(querydoc_list)):
		for j in range(len(querydoc_list[i])):
			f.write(str(querydoc_list[i][j]) + ' ')
		f.write('\n')
	f.close()

def load_querydoc(filename):
	querydoc_list = []
	querydoc_temp_list = open(filename, 'r',encoding='utf-8').readlines()
	querydoc_list = [querydoc_temp.split() for querydoc_temp in querydoc_temp_list]
	return querydoc_list

def load_label(filename):
	label_list = [int(label) for label in open(filename).readlines()]
	return label_list

def get_train_querydoc():
	return load_querydoc("dataset/segoftrain.txt")

def get_test_querydoc():
	return load_querydoc("dataset/test_querydoc.txt")

def get_age_label_list():
	return load_label("dataset/age_label_list.txt")

def get_gender_label_list():
	return load_label("dataset/gender_abel_list.txt")

def get_education_label_list():
	return load_label("dataset/eduction_label_list.txt")

def get_train_user_id_list():
	return load_label("dataset/train_user_id_list.txt")

def get_test_user_id_list():
	return load_label("dataset/test_user_id_list.txt")

def get_total_querydoc():
	train_querydoc = load_querydoc("dataset/segmentsofall.txt")
	# test_querydoc = load_querydoc("dataset/test_querydoc.txt")
	# train_querydoc.extend(test_querydoc)
	return train_querydoc
