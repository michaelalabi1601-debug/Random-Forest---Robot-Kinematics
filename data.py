import pandas as pd
import numpy as np
import matplotlib.pyplot as plt# we will use this library to show a plot of Actual vs Prediction.

df = pd.read_csv('robot_inverse_kinematics_dataset.csv')#This is the dataset we will be using for our project. It contains the joint angles and corresponding end-effector positions for a robotic arm.
print(df.isnull().sum()) # This is to check for any missing values in the dataset. It will return the count of missing values for each column.
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

#Data shapes : We print the shapes of the training and testing sets to confirm that they have been split correctly. The shape will show the number of samples and features in each set.
print(f'X_train : {X_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_train : {y_train.shape}')
print(f'y_test : {y_test.shape}')

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

# I now want to use Random Forest Hyperparameter Tuning using GridSearchCV to tune my Regressor base model.
from sklearn.model_selection import GridSearchCV
 # Define the grid of hyperparameters to search for testing
param_grid = {
    'n_estimators': [50, 100, 150], # Number of trees in the forest
    'max_depth': [None, 10, 20], # Maximum depth of the tree. a None depth allows the trees to grow fully 
    'max_features':['sqrt','log2']# Number of features to consider at each split.
    }

#Initialize the Grid Search
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2') #We initialize the GridSearchCV with our base model (rf_base), the parameter grid (param_grid), 5-fold cross-validation (cv=5), and parallel processing (n_jobs=-1) to speed up the search. The verbose parameter is set to 2 to print detailed logs of the search process.

#Run the search
print("---Starting Hyperpaarameter Tuning---")
grid_search.fit(X_train, y_train)# this is fitting the hyperparameter tuning model to the training data.
 # The best result 
print(f"Best Hyperparameters: {grid_search.best_params_}") #We print the best hyperparameters found by the grid search.

#Evaluate the best model
best_model = grid_search.best_estimator_ #We get the best model from the grid search.
y_pred_best = best_model.predict(X_test) #We use the best model to make predictions on the test set.
mse_best = mean_squared_error(y_test, y_pred_best) #We calculate the MSE for the best model.
mae_best = mean_absolute_error(y_test, y_pred_best) #We calculate the MAE for the best model.
r2_best = r2_score(y_test, y_pred_best) #We calculate the R^2 Score for the best model.
rmse_best = np.sqrt(mse_best) #We calculate the RMSE for the best model.
#Results (Best Model)
print(f"Best Model Mean Squared Error (MSE): {mse_best:.6f}") # the MSE for the best model .
print(f"Best Model Mean Absolute Error (MAE): {mae_best:.6f}") # the MAE for the best model .
print(f"Best Model R^2 Score: {r2_best:.6f}") #the R^2 Score for the best model.
print(f"Best Model Root Mean Squared Error (RMSE): {rmse_best:.6f}") # the RMSE for the best model.
