import streamlit as st 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 

#for XGBoost
import pickle
#for LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 

#dictionary of currency names
currency_model_files = {
    "AUSTRALIA - AUSTRALIAN DOLLAR/US$": {
        "type": "XGB",
        "model": "models/AUS_XGB.pkl"
    },
    "BRAZIL - REAL/US$": {
        "type": "LSTM",
        "model": "models/BRZ_LSTM.keras",
        "scaler": "models/BRZ_scaler.pkl"
    },
    "CANADA - CANADIAN DOLLAR/US$": {
        "type": "XGB",
        "model": "models/CND_XGB.pkl",
    },
    "CHINA - YUAN/US$": {
        "type": "XGB",
        "model": "models/CNY_XGB.pkl",
    },
    "DENMARK - DANISH KRONE/US$": {
        "type": "XGB",
        "model": "models/DNK_XGB.pkl"
    },
    "EURO AREA - EURO/US$": {
        "type": "XGB",
        "model": "models/EUR_XGB.pkl"
    },
    "HONG KONG - HONG KONG DOLLAR/US$": {
        "type": "LSTM",
        "model": "models/HKD_LSTM.keras",
        "scaler": "models/HKD_scaler.pkl"
    },
    "INDIA - INDIAN RUPEE/US$": {
        "type": "LSTM",
        "model": "models/IDR_LSTM.keras",
        "scaler": "models/IDR_scaler.pkl"
    },
    "JAPAN - YEN/US$": {
        "type": "XGB",
        "model": "models/JPY_XGB.pkl"
    },
    "KOREA - WON/US$": {
        "type": "XGB",
        "model": "models/KRW_XGB.pkl"
    },
    "MALAYSIA - RINGGIT/US$": {
        "type": "LSTM",
        "model": "models/MLR_LSTM.keras",
        "scaler": "models/MLR_scaler.pkl"
    },
    "MEXICO - MEXICAN PESO/US$": {
        "type": "LSTM",
        "model": "models/MXP_LSTM.keras",
        "scaler": "models/MXP_scaler.pkl"
    },
    "NORWAY - NORWEGIAN KRONE/US$": {
        "type": "XGB",
        "model": "models/NRK_XGB.pkl"
    },
    "NEW ZEALAND - NEW ZELAND DOLLAR/US$": {
        "type": "XGB",
        "model": "models/NZ_XGB.pkl"
    },
    "SOUTH AFRICA - RAND/US$": {
        "type": "LSTM",
        "model": "models/SAR_LSTM.keras",
        "scaler": "models/SAR_scaler.pkl"
    },
    "SINGAPORE - SINGAPORE DOLLAR/US$": {
        "type": "XGB",
        "model": "models/SGD_XGB.pkl"
    },
    "SRI LANKA - SRI LANKAN RUPEE/US$": {
        "type": "LSTM",
        "model": "models/SLR_LSTM.keras",
        "scaler": "models/SLR_scaler.pkl"
    },
    "SWITZERLAND - FRANC/US$": {
        "type": "XGB",
        "model": "models/SWF_XGB.pkl"
    },
    "SWEDEN - KRONA/US$": {
        "type": "XGB",
        "model": "models/SWK_XGB.pkl"
    },
    "THAILAND - BAHT/US$": {
        "type": "XGB",
        "model": "models/THB_XGB.pkl"
    },
    "TAIWAN - NEW TAIWAN DOLLAR/US$": {
        "type": "XGB",
        "model": "models/TWD_XGB.pkl"
    },
    "UNITED KINGDOM - UNITED KINGDOM POUND/US$": {
        "type": "LSTM",
        "model": "models/UK_LSTM.keras",
        "scaler": "models/UK_scaler.pkl"
    }
}

currency_column ={
    "AUSTRALIA - AUSTRALIAN DOLLAR/US$": "AUSTRALIA - AUSTRALIAN DOLLAR/US$",
    "BRAZIL - REAL/US$": "BRAZIL - REAL/US$",
    "CANADA - CANADIAN DOLLAR/US$": "CANADA - CANADIAN DOLLAR/US$",
    "CHINA - YUAN/US$": "CHINA - YUAN/US$",
    "DENMARK - DANISH KRONE/US$": "DENMARK - DANISH KRONE/US$",
    "EURO AREA - EURO/US$": "EURO AREA - EURO/US$",
    "HONG KONG - HONG KONG DOLLAR/US$": "HONG KONG - HONG KONG DOLLAR/US$",
    "INDIA - INDIAN RUPEE/US$": "INDIA - INDIAN RUPEE/US$",
    "JAPAN - YEN/US$": "JAPAN - YEN/US$",
    "KOREA - WON/US$": "KOREA - WON/US$",
    "MALAYSIA - RINGGIT/US$": "MALAYSIA - RINGGIT/US$",
    "MEXICO - MEXICAN PESO/US$": "MEXICO - MEXICAN PESO/US$",
    "NORWAY - NORWEGIAN KRONE/US$": "NORWAY - NORWEGIAN KRONE/US$",
    "NEW ZEALAND - NEW ZELAND DOLLAR/US$": "NEW ZEALAND - NEW ZELAND DOLLAR/US$",
    "SOUTH AFRICA - RAND/US$": "SOUTH AFRICA - RAND/US$",
    "SINGAPORE - SINGAPORE DOLLAR/US$": "SINGAPORE - SINGAPORE DOLLAR/US$",
    "SRI LANKA - SRI LANKAN RUPEE/US$": "SRI LANKA - SRI LANKAN RUPEE/US$",
    "SWITZERLAND - FRANC/US$": "SWITZERLAND - FRANC/US$",
    "SWEDEN - KRONA/US$": "SWEDEN - KRONA/US$",
    "THAILAND - BAHT/US$": "THAILAND - BAHT/US$",
    "TAIWAN - NEW TAIWAN DOLLAR/US$": "TAIWAN - NEW TAIWAN DOLLAR/US$",
    "UNITED KINGDOM - UNITED KINGDOM POUND/US$": "UNITED KINGDOM - UNITED KINGDOM POUND/US$"
}

def load_currency_model(currency_name):
    if currency_name not in currency_model_files: 
        raise ValueError(f'Currency {currency_name} not found in model files')
    info = currency_model_files[currency_name]
    
    # Load based on file extension
    if info['type']=='XGB':
        with open(info['model'], "rb") as f:
            model = pickle.load(f)
        return model, 'XGB', None
    elif info["type"] == "LSTM":
        model = tf.keras.models.load_model(info["model"], compile=False)
        with open(info["scaler"], "rb") as f:
            scaler = pickle.load(f)
        return model, "LSTM", scaler

#forecasting function
def forecast_xgb(
    model,
    df,
    currency_col,
    n_lags=30,
    horizon=30
):
    series = (
        pd.to_numeric(
            df[currency_col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        ).dropna().values
    )
    #defensive check
    if len(series) < 10:
        raise ValueError(f"Not enough valid data points for {currency_col}")
    history = list(series)
    preds=[]

    for _ in range(horizon):
        X_input = np.array(history[-n_lags:]).reshape(1,-1)
        y_pred = model.predict(X_input)[0]
        preds.append(y_pred)
        history.append(y_pred)

    forecast_df=pd.DataFrame({
        'Date': pd.date_range(
            start=df.index[-1]+pd.Timedelta(days=1),
            periods=horizon,
            freq='D',
        ),
        "Forecast": preds
    })
    return forecast_df

def forecast_lstm(
    model,
    df,
    currency_col,
    scaler, 
    n_input=60,
    horizon=30
):
    series = df[currency_col].values
    last_window = series[-n_input:].reshape(-1, 1)
    last_scaled = scaler.transform(last_window)
    
    preds_scaled = []
    
    for _ in range(horizon):
        X_input = last_scaled[-n_input:].reshape(1, n_input, 1)
        y_pred_scaled = model.predict(X_input)[0,0]
        preds_scaled.append(y_pred_scaled)
        last_scaled = np.append(last_scaled, y_pred_scaled).reshape(-1,1)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    
        forecast_df = pd.DataFrame({
            "Date": pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=horizon),
            "Forecast": preds
        })

    return forecast_df

#streamlit UI 
st.title('Currency Forecast APP')

#load preprocessed data 
df = pd.read_excel(
    "Foreign_Exchange_Rates.xlsx",
    sheet_name=0,
    header=None,
    engine="openpyxl"
)

# Split the single column into real columns
df = df[0].str.split(",", expand=True)

# Use first row as header
df.columns = df.iloc[0]
df = df.drop(0).reset_index(drop=True)

# Rename the date column
df = df.rename(columns={"Time Serie": "Date"})

# Convert Date to datetime and set index
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.set_index("Date")

# Convert all currency columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

#sidebar 
selected_currency = st.selectbox('Select Currency', options=list(currency_model_files.keys()))
forecast_days = st.number_input('Forecast Horizon (days)', min_value=1, max_value=365, value=30)
df_column=currency_column[selected_currency]

if st.button('Generate Forecast'):
    model, model_type,scaler = load_currency_model(df_column)
    st.write(f"Loaded model for {selected_currency}: {model_type}")

    if model_type == 'XGB':
        forecast_df = forecast_xgb(model, df, df_column, horizon=forecast_days)
    else:
        forecast_df = forecast_lstm(model, df, df_column, scaler, horizon=forecast_days)

    st.subheader('Forecast Table')
    st.dataframe(forecast_df)

    st.subheader('Forecast Chart')
    fig, ax = plt.subplots()
    ax.plot(forecast_df["Date"], forecast_df["Forecast"])
    ax.set_title("Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45)
    st.pyplot(fig)