import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta

import stock_data
import feature_engineering
import model
import visualization
import utils

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction App",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'feature_data' not in st.session_state:
    st.session_state.feature_data = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Main app title
st.title("Stock Market Prediction App")
st.markdown("Predict stock prices using machine learning models")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("Enter Alpha Vantage API Key", 
                           value="", 
                           type="password",
                           help="Get a free API key from https://www.alphavantage.co/support/#api-key")
    
    if api_key:
        st.session_state.api_key = api_key
    
    # Stock symbol input
    st.subheader("Stock Selection")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple)", value="AAPL")
    
    # Time period selection
    st.subheader("Time Period")
    lookback_period = st.slider("Historical Data Period (days)", 
                              min_value=30, 
                              max_value=365, 
                              value=180,
                              help="Amount of historical data to use")
    
    prediction_days = st.slider("Prediction Horizon (days)", 
                               min_value=1, 
                               max_value=30, 
                               value=5,
                               help="Number of days to predict into the future")
    
    # Model selection
    st.subheader("Model Configuration")
    model_type = st.selectbox("Select Model Type", 
                            options=["Linear Regression", 
                                   "Random Forest", 
                                   "SVR",
                                   "LSTM"],
                            help="Choose the machine learning algorithm")
    
    # Action buttons
    load_data_btn = st.button("Fetch Stock Data", use_container_width=True)
    train_model_btn = st.button("Train Model", use_container_width=True)

# Main content area
if not st.session_state.api_key:
    st.warning("Please enter your Alpha Vantage API key in the sidebar to get started.")
    st.info("If you don't have an API key, you can get a free one at https://www.alphavantage.co/support/#api-key")
else:
    # Load data section
    if load_data_btn or (st.session_state.data_loaded and st.session_state.selected_symbol != symbol):
        st.session_state.data_loaded = False
        st.session_state.model_trained = False
        st.session_state.selected_symbol = symbol
        
        with st.spinner(f"Fetching historical data for {symbol}..."):
            try:
                # Get stock data from API
                historical_data = stock_data.get_stock_data(st.session_state.api_key, symbol, lookback_period)
                
                if historical_data is not None and not historical_data.empty:
                    st.session_state.historical_data = historical_data
                    
                    # Generate features
                    feature_data = feature_engineering.generate_features(historical_data)
                    st.session_state.feature_data = feature_data
                    
                    st.session_state.data_loaded = True
                    st.success(f"Successfully loaded data for {symbol}")
                    
                    # Display raw data
                    with st.expander("View Raw Data"):
                        st.dataframe(historical_data)
                else:
                    st.error(f"Failed to fetch data for {symbol}. Please check if the symbol is correct.")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
    
    # Display historical chart
    if st.session_state.data_loaded:
        st.subheader(f"Historical Price Chart - {symbol.upper()}")
        historical_fig = visualization.plot_historical_data(st.session_state.historical_data)
        st.plotly_chart(historical_fig, use_container_width=True)
        
        # Display technical indicators
        st.subheader("Technical Indicators")
        indicator_fig = visualization.plot_technical_indicators(st.session_state.feature_data)
        st.plotly_chart(indicator_fig, use_container_width=True)
        
        # Train model and make predictions
        if train_model_btn:
            with st.spinner("Training model and generating predictions..."):
                try:
                    # Create, train model and make predictions
                    X_train, X_test, y_train, y_test, predictions, model_metrics = model.train_and_predict(
                        st.session_state.feature_data, 
                        model_type, 
                        prediction_days
                    )
                    
                    st.session_state.predictions = predictions
                    st.session_state.model_metrics = model_metrics
                    st.session_state.model_trained = True
                    
                    st.success("Model training complete!")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        
        # Display predictions if model is trained
        if st.session_state.model_trained:
            st.subheader("Stock Price Prediction")
            
            # Display model metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("MAE", f"{st.session_state.model_metrics['mae']:.4f}")
            with metrics_cols[1]:
                st.metric("MSE", f"{st.session_state.model_metrics['mse']:.4f}")
            with metrics_cols[2]:
                st.metric("RMSE", f"{st.session_state.model_metrics['rmse']:.4f}")
            with metrics_cols[3]:
                st.metric("R² Score", f"{st.session_state.model_metrics['r2']:.4f}")
            
            # Plot predictions
            pred_fig = visualization.plot_predictions(
                st.session_state.historical_data,
                st.session_state.predictions,
                symbol
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Feature importance (if applicable)
            if model_type in ["Random Forest", "Linear Regression"]:
                st.subheader("Feature Importance")
                importance_fig = visualization.plot_feature_importance(
                    st.session_state.model_metrics.get('feature_importance', None),
                    st.session_state.feature_data.columns
                )
                st.plotly_chart(importance_fig, use_container_width=True)

# Footer with disclaimer
st.markdown("---")
st.caption("""
**Disclaimer**: This application is for educational purposes only. The predictions are based on historical data and machine learning models,
which may not accurately predict future stock prices. Always consult with a financial advisor before making investment decisions.
""")
