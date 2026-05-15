import pandas as pd
import numpy as np
import matplotlib.pyplot as plt# we will use this library to show a plot of Actual vs Prediction.

df = pd.read_csv('robot_inverse_kinematics_dataset.csv')#This is the dataset we will be using for our project. It contains the joint angles and corresponding end-effector positions for a robotic arm.
print(df.head(10)) # For the data preview


from sklearn.model_selection import train_test_split #We will split the dataset into training and testing sets.

#Model Implementation
from sklearn.multioutput import MultiOutputRegressor #We will use this to handle our output variables 'Y' (the end-effector positions (x,y,z)).

from sklearn.ensemble import RandomForestRegressor #We will use this as our regression model to predict the Y (x,y,z) based on X (q1,q2,q3,q4,q5,q6).
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #These are the metrics we will use to evaluate our model's performance.

#Data Preprocessing
X = df[['q1','q2','q3','q4','q5','q6']] #Our input features; joint angles (q1 to q6).X is capital to show it's a matrix
y = df[['x','y','z']] #Our output variables; end-effector positions (x,y,z).y is lowercase to show it's a vector.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#We split the dataset into training and testing sets, with 20% of the data reserved for testing. The random_state parameter ensures that the split is reproducible.

#Model training (Base Model)
print("---Training Base Model---")
rf_base = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)#We initialize the Random Forest Regressor with 100 trees (n_estimators) and no maximum depth (max_depth=None). The random_state parameter ensures reproducibility.
rf_base.fit(X_train, y_train)#We fit the model to the training data (X_train and y_train).

#Model evaluation (Base Model)
print("---Evaluating Base Model---")
y_pred = rf_base.predict(X_test)#We use the trained model to make predictions on the test set (X_test).

mse = mean_squared_error(y_test, y_pred) #We calculate the Mean Squared Error (MSE) between the actual values (y_test) and the predicted values (y_pred).
mae = mean_absolute_error(y_test, y_pred) #We calculate the Mean Absolute Error (MAE) between the actual values (y_test) and the predicted values (y_pred).
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse) #We calculate the Root Mean Squared Error (RMSE) by taking the square root of the MSE using numpy's sqrt function.

#Results (Base Model)
print(f"Mean Squared Error (MSE): {mse:.6f}") #We print the MSE with 6 decimal places.
print(f"Mean Absolute Error (MAE): {mae:.6f}") #We print the MAE with 6 decimal places.
print(f"R^2 Score: {r2:.6f}") #We print the R^2 Score with 6 decimal places.
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}") #We print the RMSE with 6 decimal places.      
