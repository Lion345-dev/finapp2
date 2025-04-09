import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_ticker_data, get_sp500_tickers
from utils.visualizations import plot_candlestick
from datetime import datetime, timedelta

st.title("📈 Análisis Técnico")
st.markdown("Indicadores técnicos y patrones de velas para análisis de mercado.")

# Selección de ticker y período
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Seleccionar activo:", get_sp500_tickers(), index=0, key='ta_ticker')
with col2:
    period = st.selectbox("Período:", ['1y', '3y', '5y', '10y', 'Personalizado'], index=1)

if period == 'Personalizado':
    start_date = st.date_input("Fecha inicio:", datetime.now() - timedelta(days=365*3))
    end_date = st.date_input("Fecha fin:", datetime.now())
else:
    years = int(period[:-1])
    start_date = datetime.now() - timedelta(days=365*years)
    end_date = datetime.now()

# Cargar datos
data = load_ticker_data(ticker, start_date, end_date)

if data is not None:
    # Gráfico de velas con indicadores
    plot_candlestick(data, title=f"Análisis Técnico de {ticker}")
    
    # Selección de indicadores
    st.subheader("Indicadores Técnicos")
    indicators = st.multiselect("Agregar indicadores:", 
                              ['Media Móvil', 'Bollinger Bands', 'RSI', 'MACD', 'Volumen'])
    
    fig = go.Figure()
    
    # Gráfico de precios
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Precios',
        visible=True
    ))
    
    # Agregar indicadores seleccionados
    if 'Media Móvil' in indicators:
        window = st.slider("Período para Media Móvil:", 10, 200, 50)
        data['MA'] = data['Close'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA'],
            name=f'MA {window}d',
            line=dict(color='orange', width=2)
        ))
    
    if 'Bollinger Bands' in indicators:
        window = st.slider("Período para Bollinger Bands:", 10, 200, 20)
        std = data['Close'].rolling(window=window).std()
        data['Upper'] = data['Close'].rolling(window=window).mean() + 2*std
        data['Lower'] = data['Close'].rolling(window=window).mean() - 2*std
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Upper'],
            name='Banda Superior',
            line=dict(color='rgba(255,255,255,0.3)')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Lower'],
            name='Banda Inferior',
            line=dict(color='rgba(255,255,255,0.3)'),
            fill='tonexty'
        ))
    
    if 'RSI' in indicators:
        st.markdown("**RSI (Relative Strength Index)**")
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        
        rsi_fig.update_layout(
            title="RSI (14 días)",
            height=300,
            template="plotly_dark"
        )
        
        st.plotly_chart(rsi_fig, use_container_width=True)
    
    if 'MACD' in indicators:
        st.markdown("**MACD (Moving Average Convergence Divergence)**")
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        macd_fig = go.Figure()
        
        macd_fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            name='MACD',
            line=dict(color='blue')
        ))
        
        macd_fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Signal'],
            name='Signal Line',
            line=dict(color='orange')
        ))
        
        macd_fig.update_layout(
            title="MACD (12, 26, 9)",
            height=300,
            template="plotly_dark"
        )
        
        st.plotly_chart(macd_fig, use_container_width=True)
    
    if 'Volumen' in indicators:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volumen',
            yaxis='y2',
            marker_color='rgba(100, 100, 255, 0.3)'
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title="Volumen",
                overlaying='y',
                side='right'
            )
        )
    
    # Actualizar layout del gráfico principal
    fig.update_layout(
        title=f"Análisis Técnico de {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)