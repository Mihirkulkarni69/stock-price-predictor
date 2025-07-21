# Stock Price Predictor 📈

Predict future stock trends using yFinance + Prophet + Plotly + Streamlit.

## How to Run
```bash
streamlit run app.py

---

#### 📄 `src/predict.py`
Paste:
```python
import yfinance as yf
from prophet import Prophet
import pandas as pd

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

def train_prophet(df):
    model = Prophet()
    model.fit(df)
    return model

def predict_future(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast
