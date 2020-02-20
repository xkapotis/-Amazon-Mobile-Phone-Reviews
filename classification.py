import pandas as pd
import numpy as np

data = pd.read_csv("/media/christos/SANDISK/Data Mining/Xalkidi/Amazon_Unlocked_Mobile.csv")
reviews = pd.read_csv("./reviews_anlzd.csv")

# print(data.head())

# print(data[["Brand Name", "Price", "Rating"]])
# print(reviews["Reviews"])

data["Reviews"] = reviews["Reviews"]
# print(data.head(10))

data_for_classification = data[["Product Name", "Brand Name", "Price", "Reviews"]]
# print(data_for_classification.head(10))
# data_for_classification.to_csv(r"./data_for_classification.csv", index=None)






### Classification ###

data = pd.read_csv("./data_for_classification.csv")
# print(data.head(10))


price = data["Price"]

price_list = list(price)
price_class_table = []

for row in price_list:
    if row <= 50:
        price_class_table.append(1)
    elif row > 50 and row <= 100:
        price_class_table.append(2)
    elif row > 100 and row <= 200:
        price_class_table.append(3)
 
    elif row > 200 and row <= 500:
        price_class_table.append(4)
    elif row > 500 and row <= 1000:
        price_class_table.append(5)
    elif row > 1000 and row <= 2000:
        price_class_table.append(6)
    else :
        price_class_table.append(7)
price_list = price_class_table
price_class_table = pd.DataFrame(price_class_table)
price = price_class_table
data["Price"] = price

# print(data["Price"].unique())

data["Brand Name"] = data["Brand Name"]
data["Brand Name"] = data["Brand Name"].replace("'", "")

data["Product Name"] = data["Product Name"]
data["Product Name"] = data["Product Name"].replace("'", "")


# print(data["Brand Name"].unique())
# data = data.isnull()
# print(data.isnull())
data = data.replace(" ", np.NaN)
data = data.dropna()
# print(data)

data.to_csv(r"./data_for_classification_without_missing_values_new.csv", index=None)



# print(data.head(20))








#################### DATA for x y test and train ##############
###############################################################

# convert data["brand name"] into label beacause classifiers cant use string attributes


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
data['Brand Name']= label_encoder.fit_transform(data['Brand Name'])
data['Product Name']= label_encoder.fit_transform(data['Product Name'])  
  
# print(data['Brand Name'].unique())



data2 = data[["Product Name", "Brand Name", "Price" ]]
res2 = data['Reviews']


##############################################################
##############################################################


###################
##### Des Tree ####
###################


# from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn.model_selection import train_test_split # Import train_test_split function
# from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(data2, res2, test_size=0.3, random_state=1) # 70% training and 30% test


# ### Building Decision Tree Model ###

# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
# clf = clf.fit(X_train,y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)


# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# # accuracy 68 %








##### KNN ###########
#####################


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data2, res2, test_size=0.20)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# # Training and Predictions
# # n_neighbors osous theloume apla evala 2 epeidi toses einai oi klaseis ta result kai me 5 pernei polu kala result paromoia 
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=2)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# # Evaluating the Algorithm
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# # 


# from yellowbrick.classifier import ClassificationReport
# # Instantiate the classification model and visualizer
# visualizer = ClassificationReport(classifier)
# visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
# visualizer.score(X_test, y_test) # Evaluate the model on the test data
# g = visualizer.poof() # Draw/show/poof the data
# # de mou vzagei kalo result





#######  Log Regression ########
################################

# from sklearn.model_selection import train_test_split

# xtrain, xtest, ytrain, ytest = train_test_split( 
#         data2, res2, test_size = 0.25, random_state = 0) 

# from sklearn.preprocessing import StandardScaler 
# sc_x = StandardScaler() 
# xtrain = sc_x.fit_transform(xtrain)  
# xtest = sc_x.transform(xtest) 

# # print(xtrain)


# from sklearn.linear_model import LogisticRegression 
# classifier = LogisticRegression(random_state = 0) 
# classifier.fit(xtrain, ytrain) 


# y_pred = classifier.predict(xtest) 



# from sklearn.metrics import confusion_matrix 
# cm = confusion_matrix(ytest, y_pred) 
  
# print ("Confusion Matrix : \n", cm) 



# from sklearn.metrics import accuracy_score 
# print ("Accuracy : ", accuracy_score(ytest, y_pred)) 
# acc 74,1 %



###### Naive Bays ########
##########################

from sklearn.naive_bayes import GaussianNB
# Import train_test_split function
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data2, res2, test_size=0.3
,random_state=35
) # 70% training and 30% test


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# acc 68,5 %




###### Nn #########
###################

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data2, res2, test_size = 0.20)

# # Feature Scaling #
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# # Training and Predictions #
# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(9, 9, 9), max_iter=1000)
# mlp.fit(X_train, y_train.values.ravel())


# predictions = mlp.predict(X_test)

# # Evaluating the Algorithm #
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))

# from yellowbrick.classifier import ClassificationReport
# # Instantiate the classification model and visualizer
# visualizer = ClassificationReport(mlp)
# visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
# visualizer.score(X_test, y_test) # Evaluate the model on the test data
# g = visualizer.poof() # Draw/show/poof the data
# ##### 68 % #####




##### Random Forest #####
#########################


# # Import train_test_split function
# from sklearn.model_selection import train_test_split

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(data2, res2, test_size=0.3) # 70% training and 30% test

# #Import Random Forest Model
# from sklearn.ensemble import RandomForestClassifier

# #Create a Gaussian Classifier
# clf=RandomForestClassifier(n_estimators=20)

# #Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(X_train,y_train)

# y_pred=clf.predict(X_test)


# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# #### acc 74 %



###### SVM ########
###################

# # Import train_test_split function
# from sklearn.model_selection import train_test_split

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(data2, res2, test_size=0.3,random_state=30) # 70% training and 30% test

# #Import svm model
# from sklearn import svm

# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel

# #Train the model using the training sets
# clf.fit(X_train, y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)

# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics

# # Model Accuracy: how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# from yellowbrick.classifier import ClassificationReport
# # Instantiate the classification model and visualizer
# visualizer = ClassificationReport(clf)
# visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
# visualizer.score(X_test, y_test) # Evaluate the model on the test data
# g = visualizer.poof() # Draw/show/poof the data
### acc 




























