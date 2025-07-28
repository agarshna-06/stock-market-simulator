import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_predictions(df, predictions, confidence, ticker, model_name, model_params):
    """
    Plot historical data and predictions with confidence intervals
    
    Args:
        df (pandas.DataFrame): Historical stock data
        predictions (numpy.ndarray): Predicted values
        confidence (dict): Dictionary with lower and upper confidence bounds
        ticker (str): Stock ticker symbol
        model_name (str): Name of the model used
        model_params (dict): Model parameters
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Stock Price and Prediction", "Trading Volume")
    )
    
    # Add historical price data
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    
    # Generate future dates for predictions
    last_date = df['Date'].iloc[-1]
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, '%Y-%m-%d')
    
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Predicted Price',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(confidence['upper']) + list(confidence['lower'])[::-1],
            fill='toself',
            fillcolor='rgba(0, 176, 0, 0.2)',
            line=dict(color='rgba(0, 176, 0, 0)'),
            name='95% Confidence Interval'
        ),
        row=1, col=1
    )
    
    # Add volume as a bar chart in the second row
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker=dict(color='lightblue')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price Prediction using {model_name}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=800,
        hovermode="x unified"
    )
    
    # Add model parameters as annotations
    param_text = "Model Parameters:<br>"
    for param, value in model_params.items():
        param_text += f"{param}: {value}<br>"
    
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=param_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # Set grid style
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def plot_performance_metrics(y_true, y_pred, scaler):
    """
    Plot actual vs predicted values and the residuals
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        scaler (MinMaxScaler): Scaler used to transform the target variable
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Convert to original scale
    y_true_original = scaler.inverse_transform(y_true).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Calculate residuals
    residuals = y_true_original - y_pred_original
    
    # Create subplots: one for actual vs predicted, one for residuals
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Actual vs Predicted Values", "Residuals (Errors)")
    )
    
    # Add actual vs predicted scatter plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_true_original))),
            y=y_true_original,
            mode='lines',
            name='Actual',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_pred_original))),
            y=y_pred_original,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Add residuals plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Add zero line for residuals
    fig.add_trace(
        go.Scatter(
            x=[0, len(residuals)],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='black', dash='dash')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Set x-axis titles
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Error ($)", row=2, col=1)
    
    return fig
