from sklearn import linear_model
import numpy as np
import pandas as pd

import pickle

#Read the data into a dataframe
df = pd.read_csv("Test-results.csv", encoding="utf-8")

#Use .loc method to set X to hold the data used to predict
X = df.loc[:,["Practice1", "Practice2"]]


y = df["FinalTest"]

#Fit a linear regression model
regr = linear_model.LinearRegression()
regr.fit(X,y)

#Save the model. "wb" strand for "write binary"
pickle.dump(regr, open("model.pkl", "wb"))

#Make some sample test data
data=[[55,67]]
testdata = pd.DataFrame (data, columns=["Practice1","Practice2"])

#Predict and round results

y_pred = np.ndarray.round(regr.predict(testdata))

#Print result and round again to get rid of decimal point
print(round(y_pred[0]))