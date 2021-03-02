# Importing libraries

import pandas as pd
import pickle

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#saving the load_boston() sklearn dataset into boston
boston = load_boston()

#Creating a dataframe for the boston dataset 
df = pd.DataFrame(data = boston.data, columns = boston.feature_names )
df['price'] = boston.target

# Based on Correlation analysis, the features LSTAT, RM have strong negative and positive relationship with the target 'Price' values respectively. 
# These are important features that will have higher statistical impact on the response variable.

# PTRATIO, INDUS and TAX has moderate negative relationship with the target variable, with values around -0.51, -0.48, -0.47 respectively.


#Let's select all the features for model building

#First 13 features/input attributes are stored in X variable and the last 'price' response variable is stored in Y 
X = df.iloc[ : , 0:13]
Y = df.iloc[ : , 13]

#Splitting the data into training and testing data in 70:30 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


#Creating a multiple linear regression model using all the 12 input features. 
#The linear regression method will automatically interpret that this call is for building multiple regression model based on the number of input features passed.
#The model will automatically select the best features based on the p-values which will statistically add significance to the model.
model = LinearRegression()

#Fit the model using the training data
model.fit(X_train, Y_train)

#get the predictions for the input test data
Y_pred = model.predict(X_test)

# Evaluation of model training and testing performance
#print("Training Accuracy: ", model.score(X_train, Y_train)*100)

#print("Testing Accuracy: ", model.score(X_test, Y_test)*100)

#Using R2 to determine the goodness of the model
#print("Model Accuracy:", r2_score(Y, model.predict(X))*100)


#loading the model into a pickle object so that we can use it in flask web app
pickle_out = open('model.pkl','wb')
pickle.dump(model, pickle_out)
pickle_out.close()