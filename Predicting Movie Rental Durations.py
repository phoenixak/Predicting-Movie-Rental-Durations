import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('rental_info.csv')
# Convert to datetime format
df['rental_date'] = pd.to_datetime(df['rental_date'])
df['return_date'] = pd.to_datetime(df['return_date'])

# Calculate the number of rental days
df['rental_length_days'] = (df['return_date'] - df['rental_date']).dt.days

# Create dummy variables for "Deleted Scenes" and "Behind the Scenes"
df['deleted_scenes'] = df['special_features'].str.contains('Deleted Scenes').astype(int)
df['behind_the_scenes'] = df['special_features'].str.contains('Behind the Scenes').astype(int)

# Now, let's ensure the dummy variables are correctly created
# Check the number of ones in the 'deleted_scenes' column
print(df['deleted_scenes'].sum())

# Check the number of ones in the 'behind_the_scenes' column
print(df['behind_the_scenes'].sum())

# Define features and target
X = df.drop(['rental_length_days', 'rental_date', 'return_date', 'special_features'], axis=1)
y = df['rental_length_days']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Fit Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Get feature importances
importances = lasso.coef_

# Define hyperparameters for Random Forest
params_rf = {'n_estimators': [100, 350, 500], 'max_features': ['log2', 'auto', 'sqrt'], 'min_samples_leaf': [2, 10, 30]}

# Instantiate and fit Random Forest
rf = RandomForestRegressor(random_state=9)
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Define hyperparameters for Gradient Boosting
params_gb = {'max_depth': [2, 3, 4], 'subsample': [0.9, 1.0], 'max_features': [0.75, 1.0], 'n_estimators': [200, 300], 'random_state': [9]}

# Instantiate and fit Gradient Boosting
gb = GradientBoostingRegressor(random_state=9)
grid_gb = GridSearchCV(estimator=gb, param_grid=params_gb, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
grid_gb.fit(X_train, y_train)

# Predict using Random Forest
y_pred_rf = grid_rf.predict(X_test)

# Predict using Gradient Boosting
y_pred_gb = grid_gb.predict(X_test)

# Compute MSE for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Compute MSE for Gradient Boosting
mse_gb = mean_squared_error(y_test, y_pred_gb)

# Print MSEs
print('Random Forest MSE:', mse_rf)
print('Gradient Boosting MSE:', mse_gb)

# Save the best model
best_model = grid_rf.best_estimator_

# Save the best MSE
best_mse = mse_rf

# Print the best model and its MSE
print('Best model: Random Forest')
print('Best MSE:', best_mse)

