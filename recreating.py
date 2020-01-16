from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import numpy as np
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
import warnings

X, y = make_classification(class_sep=1.2, weights=[0.1, 0.9], n_informative=3,
                           n_redundant=1, n_features=5, n_clusters_per_class=1,
                           n_samples=10000, flip_y=0, random_state=10)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

y = y.astype("str")
y[y=="1"] = "L"
y[y=="0"] = "S"

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

X_1, X_2 = X_train[y_train=="S"], X_train[y_train=="L"]