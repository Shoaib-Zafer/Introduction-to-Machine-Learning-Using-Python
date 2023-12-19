#Now we are going to have a look on impact of changing the numbers of K- Neighbors on Accuracy of KNN Classification Algorithm 

# This line imports the load_breast_cancer function from scikit-learn, 
# which fetches the popular breast cancer dataset containing tumor features and their corresponding malignancy classifications.

from sklearn.datasets import load_breast_cancer

# This function helps us split the dataset into separate training and testing sets. 
# The training set fuels the model's learning, while the testing set assesses its performance on unseen data.

from sklearn.model_selection import train_test_split 

#This line imports the KNN classifier, a powerful tool for classifying data points based on their proximity to known examples. 
#We set n_neighbors=1, indicating the model considers only the closest neighbor when making predictions.

from sklearn.neighbors import KNeighborsClassifier

#This library provides efficient mathematical operations for manipulating numerical data like the Iris features and predictions.

import numpy as np

#This line imports the Matplotlib library with the pyplot submodule, which allows us to create and visualize plots, helpful for analyzing the impact of changing K values.

import matplotlib.pyplot as plt

#This line assigns the loaded breast cancer dataset to the variable cancer. This data contains:
#cancer.data: An array of numerical features describing each tumor.
#cancer.target: An array of labels indicating whether each tumor is malignant (1) or benign (0).

cancer = load_breast_cancer()

#This line uses the train_test_split function to split the loaded data into training and testing sets, 
#ensuring the proportion of malignant and benign tumors is preserved in both sets (stratify=cancer.target). 
#Additionally, we set a random seed (random_state=66) for reproducible results.

X_train,X_test, y_train,y_test = train_test_split(cancer.data,cancer.target, stratify=cancer.target, random_state=66)

# This line initializes an empty list (training_accuracy) to store the training accuracy for different K values.

training_accuracy=[]

#This line initializes another empty list (test_accuracy) to store the corresponding test accuracy for different K values. 

test_accuracy=[]

# This line creates a list (neighbors_count) containing integers from 1 to 10, representing the different K values we will experiment with.

neighbors_count=range(1,11)

for neighbors in neighbors_count: # This loop iterates through each K value in the neighbors_count list
    knn_clf= KNeighborsClassifier(n_neighbors=neighbors) # Inside the loop, this line creates a new KNN classifier object (knn_clf) with the current K value (neighbors) specified
    knn_clf.fit(X_train,y_train) # This line trains the KNN model on the training data (X_train features and y_train labels).
    training_accuracy.append(knn_clf.score(X_train,y_train)) # This line calculates the training accuracy for the current K value by calling the score method on the trained model with the training data. The calculated accuracy is then appended to the training_accuracy list.
    test_accuracy.append(knn_clf.score(X_test,y_test)) # Similar to the previous line, this line calculates and appends the test accuracy for the current K value using the testing data.

#this line uses Matplotlib to plot the training accuracy values (stored in training_accuracy) on the y-axis against the corresponding K values (stored in neighbors_count) 
#on the x-axis. It also adds a label "training_accuracy" to the plot for legend identification.
    
plt.plot(neighbors_count,training_accuracy, label="training_accuracy")

# Similarly, this line plots the test accuracy values (stored in test_accuracy) on the same plot with a different label ("test accuracy").

plt.plot(neighbors_count, test_accuracy, label="test accuracy")


plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.show()          

# The plot shows the training and test set accuracy on the y-axis against the setting of 
#n_neighbors on the x-axis. While real-world plots are rarely very smooth, we can still
# recognize some of the characteristics of overfitting and underfitting
#Considering a single nearest neighbor, the prediction on the training set is perfect. But when more neighbors
#are considered, the model becomes simpler and the training accuracy drops. The
#test set accuracy for using a single neighbor is lower than when using more neighbors,
#indicating that using the single nearest neighbor leads to a model that is too
#complex. On the other hand, when considering 10 neighbors, the model is too simple
#and performance is even worse. The best performance is somewhere in the middle,
# using around six neighbors. Stay Tuned
