import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_candlestick(df, title="Precios históricos"):
    """Gráfico de velas japonesas"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#2ecc71',
        decreasing_line_color='#e74c3c'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_returns_distribution(returns, title="Distribución de Retornos"):
    """Histograma de distribución de retornos"""
    fig = px.histogram(
        returns, 
        nbins=100, 
        title=title,
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        xaxis_title="Retorno Diario",
        yaxis_title="Frecuencia",
        template="plotly_dark",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_efficient_frontier(results, expected_returns, cov_matrix, risk_free_rate=0.02):
    """Frontera eficiente de Markowitz"""
    fig = go.Figure()
    
    # Portafolios aleatorios
    fig.add_trace(go.Scatter(
        x=results[1,:],
        y=results[0,:],
        mode='markers',
        name='Portafolios',
        marker=dict(
            color=results[2,:],
            colorscale='Viridis',
            showscale=True,
            size=5,
            colorbar=dict(title="Sharpe Ratio")
        )
    ))
    
    # Portafolio óptimo
    max_sharpe_idx = np.argmax(results[2])
    fig.add_trace(go.Scatter(
        x=[results[1,max_sharpe_idx]],
        y=[results[0,max_sharpe_idx]],
        mode='markers',
        name='Portafolio Óptimo',
        marker=dict(
            color='#e74c3c',
            size=12
        )
    ))
    
    fig.update_layout(
        title="Frontera Eficiente de Markowitz",
        xaxis_title="Volatilidad Anualizada",
        yaxis_title="Retorno Anualizado",
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_rolling_metrics(returns, window=21):
    """Métricas móviles: volatilidad y Sharpe ratio"""
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = (returns.rolling(window=window).mean() / returns.rolling(window=window).std()) * np.sqrt(252)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        name='Volatilidad Anualizada',
        line=dict(color='#9b59b6')
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        name='Sharpe Ratio',
        yaxis='y2',
        line=dict(color='#2ecc71')
    ))
    
    fig.update_layout(
        title=f"Métricas Móviles ({window} días)",
        xaxis_title="Fecha",
        yaxis=dict(title="Volatilidad Anualizada", side='left'),
        yaxis2=dict(title="Sharpe Ratio", side='right', overlaying='y'),
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)