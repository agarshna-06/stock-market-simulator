import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

def get_stock_data(api_key, symbol, lookback_days):
    """
    Fetches historical stock data from Alpha Vantage API
    
    Parameters:
    -----------
    api_key : str
        Alpha Vantage API key
    symbol : str
        Stock symbol (e.g., AAPL for Apple)
    lookback_days : int
        Number of days to look back for historical data
    
    Returns:
    --------
    pandas.DataFrame
        Historical stock data with columns 'date', 'open', 'high', 'low', 'close', 'volume'
    """
    try:
        # Using TIME_SERIES_DAILY endpoint for daily data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
        
        # Make API request
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception(f"API Limit: {data['Note']}")
            else:
                raise Exception("Failed to retrieve data: Unexpected API response format")
        
        # Extract time series data
        time_series = data["Time Series (Daily)"]
        
        # Convert to dataframe
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }, inplace=True)
        
        # Convert data types
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)  # Sort by date
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate lookback date
        lookback_date = datetime.now() - timedelta(days=lookback_days)
        
        # Filter data based on lookback period
        df = df[df.index >= lookback_date.strftime('%Y-%m-%d')]
        
        # Reset index and rename
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        
        return df
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except ValueError as e:
        raise Exception(f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching stock data: {str(e)}")

def get_company_info(api_key, symbol):
    """
    Fetches company information from Alpha Vantage API
    
    Parameters:
    -----------
    api_key : str
        Alpha Vantage API key
    symbol : str
        Stock symbol
    
    Returns:
    --------
    dict
        Company information
    """
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        
        return data
    except Exception as e:
        raise Exception(f"Error fetching company info: {str(e)}")
