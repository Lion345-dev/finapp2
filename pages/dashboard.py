import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_ticker_data, get_sp500_tickers, get_ticker_info
from utils.visualizations import plot_candlestick, plot_returns_distribution
from utils.calculations import calculate_returns, calculate_cagr, calculate_sharpe_ratio
from datetime import datetime, timedelta

# Configuración de la página
st.title("📊 Dashboard Financiero")
st.markdown("Análisis completo de activos financieros con visualizaciones interactivas.")

# Selección de ticker y período
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.selectbox("Seleccionar activo:", get_sp500_tickers(), index=0)
with col2:
    start_date = st.date_input("Fecha inicio:", datetime.now() - timedelta(days=365*3))
with col3:
    end_date = st.date_input("Fecha fin:", datetime.now())

# Cargar datos
data = load_ticker_data(ticker, start_date, end_date)

if data is not None:
    # Mostrar información del activo
    info = get_ticker_info(ticker)
    
    st.subheader(f"Información de {ticker}: {info['name']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sector", info['sector'])
    col2.metric("Industria", info['industry'])
    col3.metric("País", info['country'])
    
    # Gráfico de velas
    plot_candlestick(data, title=f"Precios históricos de {ticker}")
    
    # Métricas clave
    returns = calculate_returns(data['Close'])
    cagr = calculate_cagr(data['Close'])
    sharpe = calculate_sharpe_ratio(returns)
    volatility = returns.std() * np.sqrt(252)
    
    st.subheader("Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR", f"{cagr:.2f}%")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Volatilidad", f"{volatility*100:.2f}%")
    col4.metric("Último precio", f"${data['Close'].iloc[-1]:.2f}")
    
    # Distribución de retornos
    plot_returns_distribution(returns)
    
    # Datos en tabla
    st.subheader("Datos Históricos")
    st.dataframe(data.sort_index(ascending=False).style.format({
        'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 
        'Close': '{:.2f}', 'Adj Close': '{:.2f}', 'Volume': '{:,.0f}'
    }))
    