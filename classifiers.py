import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier    
from sklearn.model_selection import cross_validate
from sklearn import svm
from pprint import pprint
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, classification_report

separator = "\n ============================================================================== \n"

def split_data(x, y):
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)
   return (np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test))

def knn(x_train, y_train, x_test, y_test, verbose, unique_values):
    clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    clf.fit(x_train, y_train)
    y_pred = np.array(clf.predict(x_test))

    if verbose:
        print(separator, "KNN trained, cross-validation scores:")
    cross_val(clf, x_train, y_train)

    if verbose:
        print(separator, "Confusion matrix:")
    calc_confusion_matrix(y_test, y_pred, unique_values)


def dt( x_train, y_train, x_test, y_test, verbose, unique_values):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    if verbose:
        print(separator, "Decision-tree trained, cross-validation scores:")
    cross_val(clf, x_train, y_train)

    if verbose:
        print(separator, "Confusion matrix:")
    calc_confusion_matrix(y_test, y_pred, unique_values)



def svc(x_train, y_train, x_test, y_test, verbose, unique_values):
    clf = SVC(kernel = 'rbf', gamma = 2, C = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    if verbose:
        print(separator, "SVC trained, cross-validation scores:")
    cross_val(clf, x_train, y_train)

    if verbose:
        print(separator, "Confusion matrix:")
    calc_confusion_matrix(y_test, y_pred, unique_values)

def mlp(x_train, y_train, x_test, y_test, verbose, unique_values):
    clf = MLPClassifier(hidden_layer_sizes=(100,), 
                        alpha=1,
                        activation='relu',
                        solver = 'sgd',
                        max_iter = 3000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    if verbose:
        print(separator, "MLP trained, cross-validation scores:")
    
    cross_val(clf, x_train, y_train)

    if verbose:
        print(separator, "Confusion matrix:")
    calc_confusion_matrix(y_test, y_pred, unique_values)


def cross_val(clf, x_train, y_train):

    scoring = {
        'precision': make_scorer(precision_score, average='weighted',  zero_division=1),
        'recall': make_scorer(recall_score, average='weighted',  zero_division=1),
        'f1': make_scorer(f1_score, average='weighted', zero_division=1)
    }

    scores = cross_validate(clf, x_train, y_train, scoring=scoring, cv=5)

    pprint(scores)

def calc_confusion_matrix(y_test, y_pred, unique_values):

    print(classification_report(y_test, y_pred, target_names=unique_values, zero_division=1))

    matrix = confusion_matrix(y_test, y_pred, labels=unique_values)

    tp = matrix.diagonal()
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = matrix.sum() - (tp + fp + fn)

    print("rappel = {}".format(recall_score(y_test, y_pred, average='weighted')))
    print("precision = {}".format(precision_score(y_test, y_pred, average='weighted')))
    print("f1 = {}".format(f1_score(y_test, y_pred, average='weighted')))

    TPR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    print("True Positive Rate = {}".format(TPR))
    print("False Positive Rate = {}".format(FPR))

