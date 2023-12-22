import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/sahilrahman12/Price_prediction_of_used_Cars_-Predictive_Analysis-/master/car_data.csv')

print(df.head())

print("fuel\n",df.fuel.unique())
print("seller\n",df.seller_type.unique())
print("transmission\n",df.transmission.unique())
print("owner\n",df.owner.unique())

# Create a column new_seller_type and place it at index of the seller_type column
# Replace the values:
# Individual - 0
# Dealer - 1
# Trustmark Dealer - 2

'''x = df.seller_type.replace({"Individual":0,"Dealer":1,"Trustmark Dealer":2})

df.insert(df.columns.get_loc("seller_type"),'new_seller_type',x)
'''
# Create a column new_fuel and place it at index of the fuel column
# Replace the values:
# Petrol - 0
# Diesel - 1
# CNG - 2
# LPG - 3
# Electric - 4
'''x = df.fuel.replace({"Petrol":0,"Diesel":1,"CNG":2,"LPG":3,"Electric":4})

df.insert(df.columns.get_loc('fuel'),'new_fuel',x)
print(df)'''


#feature Engineering 
#to change categorical data to continuous i.e petrol=0,diesel=1,...etc

#fromsklearn.preprocessing import LabelEncoder
df['fuel'] = LabelEncoder().fit_transform(df['fuel'])
df['seller_type'] = LabelEncoder().fit_transform(df['seller_type'])
df['transmission'] = LabelEncoder().fit_transform(df['transmission'])
df['owner'] = LabelEncoder().fit_transform(df['owner'])


df['current_year'] = 2021
df['no_of_years'] = df['current_year'] - df['year']
print(df.head())
# Drop columns - name, year, cureent_year
# Rename selling_price to current_selling_price
df.drop(['name','year','current_year'],axis=1,inplace=True)
df.rename(columns={'selling_price':'current_selling_price'},inplace=True)

#sns.heatmap(df.corr(), annot=True, cmap='Greens')


# Select the features and targets
X = np.array(df.drop('current_selling_price',axis=1))
y = np.array(df.current_selling_price)

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Choosing the model
regressor = DecisionTreeRegressor()

# Training the model
regressor.fit(X_train, y_train)

#predict
y_pred = regressor.predict(X_test)


target = pd.DataFrame({"Actual":y_test.reshape(-1), "Predicted":y_pred.reshape(-1)})
print(target)

print(r2_score(y_test,y_pred))