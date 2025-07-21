import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta
import os


def fetch_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        raise ValueError("No data found. Try a different ticker or timeframe.")
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df.dropna(inplace=True)
    return df


def train_prophet(df: pd.DataFrame, seasonality_mode: str = "additive") -> Prophet:
    model = Prophet(seasonality_mode=seasonality_mode)
    model.add_seasonality(name="weekly", period=7, fourier_order=3)
    model.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    model.fit(df)
    return model


def predict_future(model: Prophet, days: int = 90) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Drop unnecessary columns if present
    forecast_columns = [
        "ds", "yhat", "yhat_lower", "yhat_upper", "trend", "trend_lower", "trend_upper",
        "weekly", "weekly_lower", "weekly_upper",
        "yearly", "yearly_lower", "yearly_upper",
        "additive_terms", "additive_terms_lower", "additive_terms_upper",
    ]
    forecast = forecast[[col for col in forecast_columns if col in forecast.columns]]
    return forecast


def plot_forecast(model: Prophet, forecast: pd.DataFrame):
    fig = plot_plotly(model, forecast)
    fig.update_layout(title="Stock Price Forecast", xaxis_title="Date", yaxis_title="Price")
    fig.update_traces(line=dict(color='blue'), selector=dict(name="yhat"))
    return fig


def save_forecast_to_csv(forecast: pd.DataFrame, filename="forecast.csv") -> str:
    path = os.path.join("exports", datetime.now().strftime("%Y-%m-%dT%H-%M") + "_" + filename)
    os.makedirs("exports", exist_ok=True)
    forecast.to_csv(path, index=False)
    return path
