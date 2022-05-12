#Loading the data
import pandas as pd
data0 = pd.read_csv('urldata.csv')
data0.head()

#Dropping the Domain column
data = data0.drop(['Domain'], axis = 1).copy()

#checking the data for null or missing values
data.isnull().sum()

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)
data.head()

# Sepratating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)
#X.shape, y.shape

# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

X_test.head()

#importing packages
from sklearn.metrics import accuracy_score

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
forest = RandomForestClassifier(max_depth=21)

# fit the model 
forest.fit(X_train, y_train)

# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test,forest.predict(X_test))
accuracy_score(y_test,forest.predict(X_test))

#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

y_test_forest
y_train_forest

#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

import pickle
pickle.dump(forest, open("phishing.pkl", "wb"))