from os import read
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def read_file(i):
    #path
    df_3 = "./features_3_sec.csv"
    df_30 = "./features_30_sec.csv"
    #read
    df_3 = pd.read_csv(df_3)
    df_30 = pd.read_csv(df_30)
    #preprocessing
    df_3 = df_3.dropna()
    df_30 =df_30.dropna()
    
    if i == 0:
        return df_3
    return df_30

def split_label(df):
    X = df.iloc[:, 1:len(df.columns) - 1]
    Y = df['label']
    
    lbe = LabelEncoder()
    Y = lbe.fit_transform(Y)
    
    return X, Y

def split_train_test(df, target):
    xTrain, xTest, yTrain, yTest = train_test_split(df, target, test_size=0.25, shuffle=True)
    return xTrain, xTest, yTrain, yTest

from sklearn.linear_model import LogisticRegression
def logistic_regression(df, target, xTrain, xTest, yTrain, yTest):

	lr = LogisticRegression(solver='liblinear', random_state=0)
	lr.fit(xTrain, yTrain)

	scores5 = cross_val_score(lr, xTest, yTest, cv=5)
	scores10 = cross_val_score(lr, xTest, yTest, cv=10)

	scores5 = scores5.mean()
	scores10 = scores10.mean()
	print("lr(5, 10) = (%lf, %lf)" % (scores5, scores10))
	return scores5, scores10

from sklearn.ensemble import RandomForestClassifier
def random_forest(df, target, xTrain, xTest, yTrain, yTest):

	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(xTrain, yTrain)

	scores5 = cross_val_score(clf, xTest, yTest, cv=5)
	scores10 = cross_val_score(clf, xTest, yTest, cv=10)
	scores5 = scores5.mean()
	scores10 = scores10.mean()
	print("clf(5, 10) = (%lf, %lf)" % (scores5, scores10))
	return scores5, scores10

from sklearn.naive_bayes import GaussianNB
def naive_bayes(df, target, xTrain, xTest, yTrain, yTest):
	gauss = GaussianNB()
	gauss.fit(xTrain, yTrain)

	scores5 = cross_val_score(gauss, xTest, yTest, cv=5)
	scores10 = cross_val_score(gauss, xTest, yTest, cv=10)

	scores5 = scores5.mean()
	scores10 = scores10.mean()
	print("Gauss(5, 10) = (%lf, %lf)" % (scores5, scores10))
	return scores5, scores10


def supportVectorMachine(df, target, xTrain, xTest, yTrain, yTest):
	
	#support vector machine (classification)
	svm = SVC(kernel="linear", C=100) #gamma="auto" #kernel="linear"rbf
	#svm = SVC(kernel="poly", degree=2, C=10, max_iter=10000)
	svm.fit(xTrain, yTrain)

	scores5 = cross_val_score(svm, xTest, yTest, cv=5)
	scores10 = cross_val_score(svm, xTest, yTest, cv=10)


	scores5 = scores5.mean()
	scores10 = scores10.mean()
	#result
	print("svm(5, 10) = (%lf, %lf)" % (scores5, scores10))
	return scores5, scores10


def kNN(df, target, xTrain, xTest, yTrain, yTest):

	classifier = KNeighborsClassifier(n_neighbors=3)
	classifier.fit(xTrain, yTrain)

	scores5 = cross_val_score(classifier, xTest, yTest, cv=5)
	scores10 = cross_val_score(classifier, xTest, yTest, cv=10)

	scores5 = scores5.mean()
	scores10 = scores10.mean()
	print("kNN(5, 10) = (%lf, %lf)" % (scores5, scores10))
	return scores5, scores10


def draw_graph(res5, res10):
	names = ['lr', 'clf', 'naive', 'svm', 'kNN']

	x = np.arange(len(names))
	width = 0.35

	fig, ax = plt.subplots()
	rect1 = ax.bar(x - width/2, res5, width, label='k = 5')
	rect2 = ax.bar(x + width/2, res10, width, label='k = 10')

	ax.set_ylabel('Accuracy')
	ax.set_title('Accuracy Of ' + name)
	ax.set_xticks(x)
	ax.set_xticklabels(names)
	ax.legend()

	ax.bar_label(rect1, padding=3)
	ax.bar_label(rect2, padding=3)

	fig.tight_layout()

	plt.show()
	plt.waitforbuttonpress(0)




for i in range(2):
    df = read_file(i)
    df, target = split_label(df)
    xTrain, xTest, yTrain, yTest = split_train_test(df, target)


    res5, res10 = [], []

    k5, k10 = logistic_regression(df, target, xTrain, xTest, yTrain, yTest)
    res5.append(k5 * 100)
    res10.append(k10 * 100)

    k5, k10 = random_forest(df, target, xTrain, xTest, yTrain, yTest)
    res5.append(k5 * 100)
    res10.append(k10 * 100)

    k5, k10 = naive_bayes(df, target, xTrain, xTest, yTrain, yTest)
    res5.append(k5 * 100)
    res10.append(k10 * 100)

    k5, k10 = supportVectorMachine(df, target, xTrain, xTest, yTrain, yTest)
    res5.append(k5 * 100)
    res10.append(k10 * 100)

    k5, k10 = kNN(df, target, xTrain, xTest, yTrain, yTest)
    res5.append(k5 * 100)
    res10.append(k10 * 100)


    draw_graph(res5, res10)

    print("\n\n")
