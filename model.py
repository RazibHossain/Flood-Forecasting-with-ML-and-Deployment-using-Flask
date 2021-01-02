# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing the dataset
dataset = pd.read_csv('flood.csv')
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
testdataset=pd.read_csv('testcsv.csv')
X_test=testdataset.iloc[:,:-1].values
y_test=testdataset.iloc[:,-1].values

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#confusin matrix for showing correct and incorrect 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuray_test
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

pickle.dump(classifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

    
    