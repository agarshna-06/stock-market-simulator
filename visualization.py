import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_historical_data(data):
    """
    Create a candlestick chart for historical stock data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Historical stock data with columns 'date', 'open', 'high', 'low', 'close', 'volume'
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the candlestick chart
    """
    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['volume'],
            name='Volume',
            marker_color='rgba(0, 128, 128, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Historical Stock Data',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig

def plot_technical_indicators(data):
    """
    Plot technical indicators for stock analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock data and technical indicators
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with technical indicators
    """
    # Create figure with 3 subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25])
    
    # Add price and moving averages to the first subplot
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['close'],
            name='Close Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['ma5'],
            name='5-day MA',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['ma20'],
            name='20-day MA',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['ma50'],
            name='50-day MA',
            line=dict(color='green', width=1)
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['bb_upper'],
            name='BB Upper',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['bb_lower'],
            name='BB Lower',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Add RSI to second subplot
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['rsi'],
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Add horizontal lines at 30 and 70 (overbought/oversold levels)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Add MACD to third subplot
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['macd'],
            name='MACD',
            line=dict(color='blue', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['macd_signal'],
            name='Signal Line',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['macd_hist'],
            name='MACD Histogram',
            marker_color='rgba(0, 128, 0, 0.7)'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Technical Indicators',
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    fig.update_yaxes(title_text='MACD', row=3, col=1)
    
    return fig

def plot_predictions(historical_data, predictions, symbol):
    """
    Plot historical prices and future predictions
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Historical stock data
    predictions : pandas.DataFrame
        Predicted future prices
    symbol : str
        Stock symbol
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with historical and predicted prices
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical prices
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Add predicted prices
    fig.add_trace(
        go.Scatter(
            x=predictions['date'],
            y=predictions['predicted_close'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8, symbol='diamond')
        )
    )
    
    # Highlighting the prediction range
    fig.add_vrect(
        x0=historical_data['date'].iloc[-1],
        x1=predictions['date'].iloc[-1],
        fillcolor="rgba(255, 0, 0, 0.1)",
        opacity=0.5,
        layer="below",
        line_width=0,
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        shapes=[
            # Add a vertical line at the last historical data point
            dict(
                type="line",
                xref="x", yref="paper",
                x0=historical_data['date'].iloc[-1], y0=0, 
                x1=historical_data['date'].iloc[-1], y1=1,
                line=dict(color="Black", width=1, dash="dot")
            )
        ],
        annotations=[
            dict(
                x=historical_data['date'].iloc[-1],
                y=1.05,
                xref="x",
                yref="paper",
                text="Prediction Start",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

def plot_feature_importance(importance, feature_names):
    """
    Plot feature importance from the model
    
    Parameters:
    -----------
    importance : array
        Feature importance values
    feature_names : list
        List of feature names
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with feature importance
    """
    if importance is None:
        # Return empty figure if no importance values
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importance Not Available for this Model",
            height=400
        )
        return fig
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top 15 features for readability
    importance_df = importance_df.head(15)
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis=dict(autorange="reversed")  # Put highest values at the top
    )
    
    return fig
