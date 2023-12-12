
import pandas as pd
import logging
import numpy as np
import sklearn.model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    # Source
    parm_corpusfile = './/dataset//words_type.csv'
    parm_model = './/dataset//w2v.model'
    parm_label = 'type' 
    parm_data = 'words' 

    logging.info('Data source: ' + parm_corpusfile)
    logging.info('W2v file source: ' + parm_model)
    df = pd.read_csv(parm_corpusfile, sep='\t')
    from gensim.models import word2vec
    w2vmodel = word2vec.Word2Vec.load(parm_model)
    ct = []
    for row in df[parm_data].iteritems():
        t = []
        for word in row[1].split(' '):
            if word in w2vmodel.wv.key_to_index:
                t.append(w2vmodel.wv[word])
        ct.append(np.mean(np.array(t), axis=0).tolist())

    df['w2v'] = ct
    df = df.dropna()  # delete null line
    logging.info('Amount: %d , Positive: %d , Negative: %d ' %(len(df),len(df[df['type']==1]),len(df[df['type']==-1])))

    x_train = df['w2v'].tolist()
    label = df['type'].tolist()

    # model = MultinomialNB()
    # accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    # print('MultinomialNB 交叉验证结果:', accs, '均值：', accs.mean())

    model = GaussianNB()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    print('GaussianNB 交叉验证结果:', accs, '均值：', accs.mean())
    #
    model = LogisticRegression()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    print('LogisticRegression 交叉验证结果:', accs,'均值：', accs.mean())

    model = LinearSVC()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    print('LinearSVC 交叉验证结果:', accs, '均值：', accs.mean())