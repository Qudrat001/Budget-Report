# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# ------------------------------
# 5. MACHINE LEARNING (Regression Model)
# ------------------------------

def train_regression_model(file_path):
    df = pd.read_csv(file_path)
    
    # Features and target variable
    X = df[['Advertising Budget ($)', 'Social Media Spend ($)', 'Influencer Budget ($)']]
    y = df['Sales Revenue ($)']
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f'MAE: {mae}, RMSE: {rmse}')
    
    return model
