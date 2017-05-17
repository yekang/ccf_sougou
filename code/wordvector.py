from sklearn.feature_extraction.text import CountVectorizer
import data_preprocess 
def tovector():
    ctv=CountVectorizer()
    train_data=data_preprocess.get_train_querydoc()
    text_data_train = ctv.fit_transform('train')
    print(text_data_train.shape)
    # return text_data_train

if __name__ == '__main__':
    tovector()
    # print(text_data_train)