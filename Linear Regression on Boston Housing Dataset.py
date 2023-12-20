import mglearn # Provides datasets and visualizations for machine learning

# The below function helps us split the dataset into separate training and testing sets. 
# The training set fuels the model's learning, while the testing set assesses its performance on unseen data.

from sklearn.model_selection import train_test_split

#The model represents the relationship as a straight line (for simple linear regression) or a hyperplane (for multiple linear regression), expressed as:

#y = b0 + b1x1 + b2x2 + ... + bnxn
#y: dependent variable (predicted value)
#b0: intercept (value of y when all x's are 0)
#bi: coefficients (slopes) for each independent variable

from sklearn.linear_model import LinearRegression


#Boston Housing dataset has 506 samples and 105 derived features the target is median house price which we want to predict based on these features
 
X,y = mglearn.datasets.load_extended_boston()

X_train,X_test,y_train,y_test= train_test_split(X,y, random_state=0) # Divides the data into training and testing sets. random_state Ensures reproducibility by fixing the random seed, generating consistent splits across multiple runs.

print("Data Type of X_train:{}".format(type(X_train)))

print("Data Type of y_train:{}".format(type(y_train) ))

print("Numer of Features and Data Points in X_train:{}".format(X_train.shape)) # Reveal the structure of the data, aiding in understanding its organization and dimensions

print("Numer of Data Points in y_train:{}".format(y_train.shape)) # Reveal the structure of the data, aiding in understanding its organization and dimensions


lr= LinearRegression().fit(X_train, y_train) # Instantiates a linear regression model object.Trains the model using the training data, uncovering patterns and relationships between features and target.
lr_coefficients= lr.coef_ # Captures the learned coefficients (slopes) for each feature, quantifying their impact on predictions.
lr_intercept=lr.intercept_ #  Represents the intercept term, the predicted value when all features are zero

print("Data Type of Coefficients Returned by Linear Regrssion:{}".format(type(lr_coefficients)))
print("Number of Coefficients retuned by Linear Regression:{}".format(lr_coefficients.shape))

print("Intercepts Returned by Linear Regrssion:{}".format(lr_intercept))

print("Training set Accuracy: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set Accuracy: {:.2f}".format(lr.score(X_test, y_test)))

# The observed discrepancy between training and test set performance indicates overfitting in our linear regression model. 
# To address this issue and control model complexity, we should explore alternative approaches like ridge regression.