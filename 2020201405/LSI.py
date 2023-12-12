import pandas as pd
import logging

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    corpusfile = './/dataset//words_type.csv'
    label = 'type'
    data = 'words'

    df = pd.read_csv(corpusfile,sep='\t')
    label_ = df[label]
    data_ = df[data]

    #tfidf模型
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 4),
        # max_features=150000
    )
    # tfidf.fit(data_)
    x_train = tfidf.fit_transform(data_)

    lsa = TruncatedSVD(n_components=300, n_iter=10, random_state=42)
    # lsa.fit(x_train)
    x_train = lsa.fit_transform(x_train)

    model = GaussianNB()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label_, scoring='precision', cv=10, n_jobs=1)
    print('GaussianNB 交叉验证结果:', accs,'均值：', accs.mean())

    model = LogisticRegression()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label_, scoring='precision', cv=10, n_jobs=1)
    print('LogisticRegression 交叉验证结果:', accs,'均值：', accs.mean())

    model = LinearSVC()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label_, scoring='precision', cv=10, n_jobs=1)
    print('LinearSVC 交叉验证结果:', accs, '均值：', accs.mean())