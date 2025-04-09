import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)
def load_ticker_data(ticker, start_date, end_date):
    """Cargar datos históricos de Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No se encontraron datos para {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

def get_sp500_tickers():
    """Obtener lista de tickers del S&P500"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].tolist()
    except:
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM', 'V']

def get_ticker_info(ticker):
    """Obtener información básica de un ticker"""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Desconocido'),
            'industry': info.get('industry', 'Desconocido'),
            'country': info.get('country', 'Desconocido'),
            'marketCap': info.get('marketCap', 0),
            'description': info.get('longBusinessSummary', 'No hay descripción disponible.')
        }
    except:
        return {
            'name': ticker,
            'sector': 'Desconocido',
            'industry': 'Desconocido',
            'country': 'Desconocido',
            'marketCap': 0,
            'description': 'No hay información disponible.'
        }