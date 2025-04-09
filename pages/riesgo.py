import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import load_ticker_data, get_sp500_tickers
from utils.calculations import (
    calculate_returns, 
    calculate_annualized_volatility,
    calculate_var,
    calculate_beta,
    calculate_capm
)
from datetime import datetime, timedelta

st.title("📉 Medición de Riesgo")
st.markdown("Herramientas avanzadas para evaluar el riesgo de activos financieros.")

# Selección de ticker y benchmark
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Seleccionar activo:", get_sp500_tickers(), index=0, key='risk_ticker')
with col2:
    benchmark = st.selectbox("Benchmark (para Beta y CAPM):", ['^GSPC (S&P500)', '^IXIC (NASDAQ)', '^DJI (Dow Jones)'], index=0)

# Período de análisis
years = st.slider("Años de histórico:", 1, 10, 5)
start_date = datetime.now() - timedelta(days=365*years)
end_date = datetime.now()

# Cargar datos
ticker_data = load_ticker_data(ticker, start_date, end_date)
benchmark_data = load_ticker_data(benchmark.split()[0], start_date, end_date)

if ticker_data is not None and benchmark_data is not None:
    # Calcular retornos
    ticker_returns = calculate_returns(ticker_data['Close'])
    benchmark_returns = calculate_returns(benchmark_data['Close'])
    
    # Métricas de riesgo
    volatility = calculate_annualized_volatility(ticker_returns)
    var_95 = calculate_var(ticker_returns, 0.95)
    var_99 = calculate_var(ticker_returns, 0.99)
    beta = calculate_beta(ticker_returns, benchmark_returns)
    capm_return = calculate_capm(ticker_returns, benchmark_returns)
    
    # Mostrar métricas
    st.subheader("Métricas de Riesgo")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatilidad Anualizada", f"{volatility*100:.2f}%")
    col2.metric("VaR 95%", f"{var_95:.2f}%")
    col3.metric("Beta", f"{beta:.2f}")
    col4.metric("Retorno CAPM", f"{capm_return*100:.2f}%")
    
    # Gráfico de distribución de retornos con VaR
    fig = px.histogram(ticker_returns*100, nbins=100, title="Distribución de Retornos con VaR")
    
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange", annotation_text=f"VaR 95%: {var_95:.2f}%")
    fig.add_vline(x=var_99, line_dash="dash", line_color="red", annotation_text=f"VaR 99%: {var_99:.2f}%")
    
    fig.update_layout(
        xaxis_title="Retorno Diario (%)",
        yaxis_title="Frecuencia",
        template="plotly_dark",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling volatility
    st.subheader("Volatilidad Móvil (21 días)")
    rolling_vol = ticker_returns.rolling(window=21).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        name='Volatilidad',
        line=dict(color='#3498db')
    ))
    
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Volatilidad Anualizada (%)",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de dispersión para Beta
    st.subheader(f"Análisis Beta vs {benchmark}")
    
    fig = px.scatter(
        x=benchmark_returns*100,
        y=ticker_returns*100,
        trendline="ols",
        labels={'x': f"Retorno {benchmark} (%)", 'y': f"Retorno {ticker} (%)"},
        title=f"Beta = {beta:.2f}"
    )
    
    fig.update_layout(
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress testing
    st.subheader("Pruebas de Estrés")
    stress_periods = {
        'Crisis COVID (2020)': ('2020-01-01', '2020-06-30'),
        'Crisis Financiera (2008)': ('2008-01-01', '2009-12-31'),
        'Caída Bitcoin (2018)': ('2018-01-01', '2018-12-31')
    }
    
    selected_stress = st.selectbox("Seleccionar período de estrés:", list(stress_periods.keys()))
    
    stress_start, stress_end = stress_periods[selected_stress]
    stress_data = load_ticker_data(ticker, stress_start, stress_end)
    
    if stress_data is not None:
        stress_returns = calculate_returns(stress_data['Close'])
        stress_drawdown = (stress_data['Close'] / stress_data['Close'].cummax() - 1) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Máximo Drawdown", f"{stress_drawdown.min():.2f}%")
        col2.metric("Volatilidad Anualizada", f"{stress_returns.std()*np.sqrt(252)*100:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stress_data.index,
            y=stress_data['Close'],
            name='Precio',
            line=dict(color='#e74c3c')
        ))
        
        fig.update_layout(
            title=f"Comportamiento durante {selected_stress}",
            xaxis_title="Fecha",
            yaxis_title="Precio",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)