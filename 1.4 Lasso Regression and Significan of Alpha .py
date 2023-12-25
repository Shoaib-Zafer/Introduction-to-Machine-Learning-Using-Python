
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso

import mglearn

import  numpy as np

X,y = mglearn.datasets.load_extended_boston()

X_train,X_test,y_train,y_test= train_test_split(X,y, random_state=0)

lasso= Lasso().fit (X_train,y_train)
print ("Accuracy of Lasso Regression on Training Data with Alpha = 1.0:", lasso.score(X_train,y_train))
print ("Accuracy of Lasso Regression on Test Data with Alpha = 1.0:", lasso.score(X_test,y_test))
print("Number of Features Used in this regression:", np.sum(lasso.coef_ !=0), "\n")

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set Lasso Accuracy score for Alpha= .01:",lasso001.score(X_train, y_train))
print("Test set Lasso Accuracy Score for Aplpa=.01",lasso001.score(X_test, y_test))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)),"\n")

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set Lasso Accuracy score for Alpha= .0001:",lasso00001.score(X_train, y_train))
print("Test set Lasso Accuracy score for Alpha= .0001:",lasso00001.score(X_test, y_test))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))