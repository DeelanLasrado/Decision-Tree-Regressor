import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor


#problem statement
'''In this data, we have one independent variable 'Temperature' and one dependent variable 'Revenue'.
 You have to build a DecisionTreeRegressor to study the relationship b/w the two variables  of the Ice Cream Shop 
 and then predict the revenue for the ice cream shop based on the temperature on a particular day.'''


df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')

print(df.head())
print(df.isnull().sum())

X = np.array(df.Temperature.values)
y = np.array(df.Revenue.values)


#machine learning
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
regressor = DecisionTreeRegressor()

regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = regressor.predict(X_test.reshape(-1,1))
print(y_test)
print()
print(y_pred)

#performance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print(r2_score(y_test,y_pred))