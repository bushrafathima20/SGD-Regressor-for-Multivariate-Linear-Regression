# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate input and output variables.

2. Split the data into training and testing sets.

3. Train the linear regression model using the training data.

4. Predict the output for the test data and evaluate the results.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: BUSHRA FATHIMA I
RegisterNumber: 212225040051
*/

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("")
data.columns = data.columns.str.strip()
X = data[['Size', 'Bedrooms']]

y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:
<img width="380" height="49" alt="Screenshot 2026-04-27 143549" src="https://github.com/user-attachments/assets/abdb72fa-7762-46a2-83ff-efdbb7c40aa6" />
<img width="1307" height="216" alt="Screenshot 2026-04-27 144528" src="https://github.com/user-attachments/assets/38e5cbd1-14a0-41f8-b761-669a03635658" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
