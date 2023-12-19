# This line fetches the popular Iris dataset from scikit-learn, containing 150 flower specimens described by four features and belonging to three distinct species

from sklearn.datasets import load_iris

# This function helps us split the dataset into separate training and testing sets. 
# The training set fuels the model's learning, while the testing set assesses its performance on unseen data.

from sklearn.model_selection import train_test_split 

#This line imports the KNN classifier, a powerful tool for classifying data points based on their proximity to known examples. 
#We set n_neighbors=1, indicating the model considers only the closest neighbor when making predictions.

from sklearn.neighbors import KNeighborsClassifier

#This library provides efficient mathematical operations for manipulating numerical data like the Iris features and predictions.
import numpy as np

# 1.0 Exploring IRIS Dataset 

iris_dataset = load_iris() # Loads IRIS data in the variable iris_dataset 

# These two lines reveal the available information within the dataset and provide a comprehensive description of its contents, 
# including data collection details and attribute definitions.

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys())) 
print(iris_dataset['DESCR'] + "\n...") 

 # These two lines unveil the names associated with each class label (e.g., Setosa, Versicolor, Virginica) and the four features used to describe each data point 
 # (e.g., sepal length, petal width).

print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))

# These two lines confirm that the data itself is stored as a NumPy array, revealing its size (150 samples with 4 features).

print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape)) 

# This showcases the first few data points, demonstrating the format of individual entries (numerical values representing the four features for each flower).

print("First five columns of data:\n{}".format(iris_dataset['data'][:5])) 

# These two lines clarify that the target labels (specifying the species) are also stored as a NumPy array with the same size as the data, 
# ensuring a one-to-one correspondence between data points and their classes.

print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))

# This displays the actual class labels for each data point, represented as numerical indices (0, 1, or 2) corresponding to the species names 

print("Target:\n{}".format(iris_dataset['target']))

# 2.0 Splitting Train and Test Data:

# This line utilizes the train_test_split function to partition the dataset into training and testing sets. 
# We provide both features and target labels as input, and setting random_state=0 ensures consistent splitting across program runs.

X_train, X_test , y_train, y_test = train_test_split (iris_dataset['data'], iris_dataset['target'], random_state=0)

# These lines print the sizes of the resulting training and testing sets for both features and labels. 
# This reveals the data distribution allocated for training the model and evaluating its performance.

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# 3.0 Building and Training the KNN Classifier:

# This line creates a KNN classifier with n_neighbors=1, specifying that the model should consider only the closest data point (its first neighbor) when making predictions. 
# This simplifies the logic while still providing effective classification for the Iris dataset.

knn = KNeighborsClassifier(n_neighbors=1)

# This function trains the KNN model using the labeled training data. It analyzes the feature values and their corresponding class labels,
#  learning the underlying relationships and patterns within the data.

knn.fit(X_train, y_train)

# 4.0 Making Predictions 

# This variable defines a new data point as a NumPy array, representing an unseen Iris flower you want to classify. 
# It could be based on actual measurements of a new flower or fabricated for testing purposes.

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

# This function applies the trained KNN model to the new data point X_new. 
# It analyzes the features of X_new and compares them to all training data points. Based on the closest neighbor with n_neighbors=1, 
# the KNN model predicts the class of the new data point.

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))

# This translates the numerical prediction into the corresponding species name (Setosa, Versicolor, or Virginica),
#  providing a human-readable interpretation of the KNN model's classification.

print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# This line uses the trained KNN model to predict the classes of all data points in the testing set (X_test). 
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# This function calculates the KNN model's accuracy on the testing set.
#  It compares the predicted class labels (y_pred) with the actual class labels (y_test) and returns the percentage of correct predictions.
#   This score provides an objective measure of the model's generalization ability on unseen data.

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# This KNN Model has got accuracy of 97 %