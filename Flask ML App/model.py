# Importing necessary libraries
import pandas as pd # To manage data as data frames
import numpy as np # To manipulate data as arrays
from sklearn.neighbors import KNeighborsClassifier # Classification model
data = pd.read_csv('iris.csv')

# Dictionary containing the mapping
variety_mapping = {0:'Setosa',1:'Versicolor',2:'Virginica'}

#Encoding the target variable to numeric
data = data.replace(['Setosa','Versicolor','Virginica'],[0,1,2])


X = data.iloc[:, 0:-1] # Extracting the features/independent variables
y = data.iloc[:, -1] # Extracting the target/dependent variable

knn = KNeighborsClassifier(n_neighbors=5) #Initializing knn Model
knn.fit(X,y) # Fitting the model

# Function for classification based on inputs
def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mapping[knn.predict(query)[0]] # Retrieve from dictionary
    return prediction # Return the prediction
