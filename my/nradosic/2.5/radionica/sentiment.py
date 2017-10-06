import re
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    return text


def tokenizer(text):
    return text.split()


def learning(lr_tfidf, X_train, y_train):
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=lr_tfidf,
                       X=X_train,
                       y=y_train,
                       train_sizes=np.linspace(0.1, 1.0, 10),
                       cv=5,
                       n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()


def validation(lr_tfidf, X_train, y_train):
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
        estimator=lr_tfidf,
        X=X_train,
        y=y_train,
        param_name='clf__C',
        param_range=param_range,
        cv=5)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1.0])
    plt.tight_layout()
    plt.show()


def confusion(lr_tfidf, X_train, y_train, X_test, y_test):
    lr_tfidf.fit(X_train, y_train)
    y_pred = lr_tfidf.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('./movie_data.csv', encoding='utf-8')
    df.head(3)

    X = df.loc[:, 'review'].apply(preprocessor).values
    y = df.loc[:, 'sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None,
                            ngram_range=(1, 1),
                            tokenizer=tokenizer,
                            norm='l2')

    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0, penalty='l2', C=10.0))])

    scores = cross_val_score(estimator=lr_tfidf,
                             X=X_train,
                             y=y_train,
                             cv=5)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    label = {0: 'negative', 1: 'positive'}
    example = []
    with open('probni.txt', 'r') as file:
        example.append(file.readline())

    lr_tfidf.fit(X_train, y_train)
    print('Prediction: %s\nProbability: %.2f%%' % \
          (label[lr_tfidf.predict(example)[0]], lr_tfidf.predict_proba(example).max() * 100))

    p1 = Process(name='daemon-learning_curve', target=learning, args=(lr_tfidf, X_train, y_train))
    p1.daemon = True
    p2 = Process(name='daemon-validation_curve', target=validation, args=(lr_tfidf, X_train, y_train))
    p2.daemon = True
    p3 = Process(name='daemon-confusion_matrix', target=confusion, args=(lr_tfidf, X_train, y_train, X_test, y_test))
    p3.daemon = True

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
