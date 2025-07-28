import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_predict(feature_data, model_type, prediction_days):
    """
    Train a machine learning model and predict future stock prices
    
    Parameters:
    -----------
    feature_data : pandas.DataFrame
        DataFrame with features and target variable
    model_type : str
        Type of model to use ('Linear Regression', 'Random Forest', 'SVR', or 'LSTM')
    prediction_days : int
        Number of days to predict into the future
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, predictions, metrics)
    """
    # Make a copy of the data to avoid modifying the original
    df = feature_data.copy()
    
    # Drop non-feature columns
    non_feature_cols = ['date', 'day_of_week', 'month', 'quarter']
    feature_cols = [col for col in df.columns if col not in non_feature_cols and col != 'target']
    
    # Handle missing values
    df = df.fillna(method='ffill')
    
    # Prepare data for training
    X = df[feature_cols].values
    y = df['target'].values
    
    # Drop the last row since target is NaN
    X = X[:-1]
    y = y[:-1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Normalize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target variable for better model performance
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Initialize dictionary for metrics
    model_metrics = {}
    
    # Train model based on selected type
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Store feature importance
        model_metrics['feature_importance'] = model.coef_
        
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions (no need to scale target for tree-based models)
        y_pred = model.predict(X_test_scaled)
        
        # Store feature importance
        model_metrics['feature_importance'] = model.feature_importances_
        
    elif model_type == "SVR":
        model = SVR(kernel='rbf')
        model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
    elif model_type == "LSTM":
        # Reshape input for LSTM [samples, time steps, features]
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compile and train
        model.compile(optimizer='adam', loss='mse')
        model.fit(
            X_train_lstm, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.1,
            verbose=0
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_lstm)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    model_metrics.update({
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    })
    
    # Make future predictions
    future_predictions = []
    
    # Get the last data point (which would be used to predict the next day)
    last_features = df[feature_cols].values[-1].reshape(1, -1)
    last_features_scaled = scaler_X.transform(last_features)
    
    # For LSTM, reshape
    if model_type == "LSTM":
        last_features_scaled = last_features_scaled.reshape(1, 1, last_features_scaled.shape[1])
    
    # Make predictions for the next 'prediction_days' days
    for i in range(prediction_days):
        if model_type == "LSTM":
            next_pred_scaled = model.predict(last_features_scaled)
            next_pred = scaler_y.inverse_transform(next_pred_scaled)[0][0]
        else:
            next_pred_scaled = model.predict(last_features_scaled)
            if model_type in ["Linear Regression", "SVR"]:
                next_pred = scaler_y.inverse_transform(next_pred_scaled.reshape(-1, 1))[0][0]
            else:
                next_pred = next_pred_scaled[0]
        
        future_predictions.append(next_pred)
        
        # Update last features for the next prediction
        # (This is a simplification - in a real scenario we would update all features)
        # Just shift the close price for this example
        last_features[0, 3] = next_pred  # assuming 'close' is at index 3
        last_features_scaled = scaler_X.transform(last_features)
        if model_type == "LSTM":
            last_features_scaled = last_features_scaled.reshape(1, 1, last_features_scaled.shape[1])
    
    # Create future dates for the predictions
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=prediction_days, freq='D')
    
    # Combine into a dataframe
    predictions_df = pd.DataFrame({
        'date': future_dates,
        'predicted_close': future_predictions
    })
    
    return X_train, X_test, y_train, y_test, predictions_df, model_metrics
