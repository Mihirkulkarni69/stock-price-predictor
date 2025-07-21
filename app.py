import streamlit as st
from src.predict import (
    fetch_data,
    train_prophet,
    predict_future,
    plot_forecast,
    save_forecast_to_csv,
)

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Trend Predictor")

# Sidebar Inputs
st.sidebar.header("1. Select Stock Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, TSLA)", "AAPL")
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)

st.sidebar.header("2. Advanced Options")
with st.sidebar.expander("🔧 Show Advanced Options"):
    seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"], index=0)
    export_csv = st.checkbox("Export forecast to CSV", value=True)

# Main Logic
if st.button("🔮 Predict"):
    try:
        with st.spinner("Fetching and preparing data..."):
            df = fetch_data(ticker, period, interval)
            st.success("✅ Data fetched successfully!")

        st.write("### Raw Data Preview", df.tail())

        with st.spinner("Training model and generating forecast..."):
            model = train_prophet(df, seasonality_mode=seasonality_mode)
            forecast = predict_future(model, forecast_days)

        st.write("### Forecast Output")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days))

        st.write("### Forecast Plot")
        fig = plot_forecast(model, forecast)
        st.plotly_chart(fig)


        if export_csv:
            path = save_forecast_to_csv(forecast)
            st.success(f"📁 Forecast exported to: `{path}`")
            with open(path, "rb") as f:
                st.download_button("Download CSV", f, file_name="forecast.csv")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
