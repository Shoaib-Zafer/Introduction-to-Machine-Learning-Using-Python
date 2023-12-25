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

# Imports Ridge regression library 

from sklearn.linear_model import Ridge


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

print("Training set Accuracy of Linear Regression: {:.2f}".format(lr.score(X_train, y_train))) # Computes Accuracy of the predictions in Training Data
print("Test set Accuracy of Linear Regression : {:.2f}".format(lr.score(X_test, y_test))) # # Computes Accuracy of the predictions in Test Data

# The observed discrepancy between training and test set performance indicates overfitting in our linear regression model. 
# To address this issue and control model complexity, we should explore alternative approaches like ridge regression.

#Ridge regression is a regularization technique applied within linear regression frameworks to mitigate overfitting and enhance model generalizability. 
#While it employs the same prediction formula as ordinary least squares, it introduces an L2 regularization term to the loss function. 
#This term penalizes large coefficient values, effectively constraining model complexity and promoting coefficient shrinkage towards zero.

#The objective of ridge regression is to strike an optimal balance between predictive performance on the training data and model simplicity.
# By attenuating the influence of individual features, it reduces the risk of overfitting and fosters models that generalize more effectively to unseen data. 
# This makes ridge regression a valuable tool for constructing robust and dependable predictive models, particularly in scenarios where feature multicollinearity 
# or high-dimensionality pose challenges.

ridge_model=Ridge().fit(X_train,y_train) #Instantiates a Ridge regression model object.Trains the model using the training data, uncovering patterns and relationships between features and target.

print("Training set Accuracy of Ridge Regression with Apha =1.0 : {:.2f}".format(ridge_model.score(X_train, y_train))) # Computes Accuracy of the predictions in Training Data
print("Test set Accuracy of Ridge Regression with Alpha= 1.0 : {:.2f}".format(ridge_model.score(X_test, y_test))) # Computes Accuracy of the predictions in Training Data

# The observed discrepancy in scores lower on the training set but higher on the test set affirms Ridge regression's effectiveness in combating overfitting. 
# While its restrained nature might slightly compromise performance on training data, it yields superior generalization to unseen data, a hallmark of robust predictive models. 
# This preference for generalization cements Ridge regression as the model of choice over LinearRegression in this context.

# How Alpha Impact Ridge Regression 

# The Ridge model balances the simplicity of the model (few non-zero coefficients) with its accuracy on the training data. 
# The user can control the balance between simplicity and training accuracy using the alpha parameter. 
# The default value of alpha is 1.0, but this may not be the optimal setting for a particular dataset. 
# Increasing alpha forces the coefficients to be closer to zero, which improves generalization but may reduce training accuracy.

# now we run the ridge regression for different values of Alpha and compare the accuracy results 


ridge_model10= Ridge(alpha=10).fit(X_train,y_train) # Here we set value of Alpha =10 which will force the coefficients to become close to zero 
print("Training Set Accuracy of Ridge Regression with Alpha= 10 : {:.2f}".format (ridge_model10.score(X_train,y_train)))
print("Testing Set Accuracy of Ridge Regression with Alpha= 10 : {:.2f}".format (ridge_model10.score(X_test,y_test)))     

# performance of the model has decreased by increasing the value of Alpha 
# Reducing the alpha parameter loosens the constraints on the coefficients. When alpha is very small, the coefficients are barely restricted, 
# resulting in a model similar to LinearRegression.

ridge_model01= Ridge(alpha=0.1).fit(X_train,y_train) # Here we set value of Alpha =0.1 
print("Training Set Accuracy of Ridge Regression with Alpha= 0.1 : {:.2f}".format (ridge_model01.score(X_train,y_train)))
print("Testing Set Accuracy of Ridge Regression with Alpha= 0.1 : {:.2f}".format (ridge_model01.score(X_test,y_test)))   


# Here, alpha=0.1 seems to be working well. We could try decreasing alpha even more to improve generalization