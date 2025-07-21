# 📈 Stock Price Predictor

An interactive web app built with **Streamlit** and **Prophet** to forecast future stock prices. Upload your own stock CSV file or select a ticker to visualize trends and get AI-powered future price predictions.

---

## 🚀 Features

- 🔍 Fetch live stock data using **Yahoo Finance**
- 📅 Predict future prices using **Facebook Prophet**
- 📊 Interactive visualizations with **Plotly**
- 📂 Export forecasts to CSV
- ⚙️ Advanced customization options (seasonality, changepoints, etc.)
- 🧠 Powered by machine learning

---

## 📷 Demo

> _Coming soon!_ (You can add a Streamlit Cloud/Render/Hugging Face deployment link here)

---

## 🧑‍💻 How to Run Locally

### ✅ Prerequisites

- Python 3.8+
- pip
- Git

### 🔧 Setup

```bash
# 1. Clone the repository
git clone https://github.com/mihirkulkarni69/stock-price-predictor.git
cd stock-price-predictor

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
