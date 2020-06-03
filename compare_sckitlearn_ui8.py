# -*- coding: utf-8 -*-
"""
@author: umit isikdag -2020
"""

import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import seaborn as sn

import warnings
warnings.filterwarnings("ignore")

seed = 1
from sklearn.preprocessing import LabelEncoder
dataset = pandas.read_csv('data_experiment_2_Weka.csv')
X = dataset.iloc[:, 0:5].values
Y = dataset.iloc[:, 5].values

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

models = []
models.append(('KNN', KNeighborsClassifier(4)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel="rbf", C=0.025, probability=True,gamma='scale')))
models.append(('RF',RandomForestClassifier(n_estimators=100)))
models.append(('AB',AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=500,learning_rate=1)));
models.append(('GB',GradientBoostingClassifier()))

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    kfold2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X,encoded_Y, cv=kfold, scoring=scoring)
    cv_results2 =cross_val_score(model, X,encoded_Y, cv=kfold2, scoring=scoring)
    y_pred = cross_val_predict(model, X,encoded_Y,cv=kfold)
    conf_mat = confusion_matrix(encoded_Y, y_pred)
    results.append(cv_results)
    names.append(name)

    print('-------------Classification Report--------------')
    print(classification_report(encoded_Y, y_pred, digits=3))
    #Prepare Confusion Matrices
    df_cm = pandas.DataFrame(conf_mat, columns=np.unique(encoded_Y), index = np.unique(encoded_Y))
    df_cm.index.name = 'True/Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 40},fmt='d')# font size
    #fix to seaborn figure
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() # show the plot

#Feature Importaces
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, encoded_Y)
fi1=clf.feature_importances_

clf2=AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=500,learning_rate=1)
clf2 = clf2.fit(X, encoded_Y)
fi2=clf2.feature_importances_

clf3=GradientBoostingClassifier()
clf3 = clf3.fit(X, encoded_Y)
fi3=clf3.feature_importances_

clf4=DecisionTreeClassifier()
clf4 = clf4.fit(X, encoded_Y)
fi4=clf4.feature_importances_


