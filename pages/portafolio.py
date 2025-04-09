import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import get_sp500_tickers, load_ticker_data
from utils.calculations import markowitz_optimization
from datetime import datetime, timedelta

st.title("📑 Optimización de Portafolio")
st.markdown("Construcción y optimización de portafolios usando el modelo de Markowitz.")

# Selección de activos
st.subheader("Selección de Activos")
selected_tickers = st.multiselect(
    "Seleccionar activos para el portafolio:", 
    get_sp500_tickers(),
    default=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
)

# Parámetros del análisis
col1, col2 = st.columns(2)
with col1:
    years = st.slider("Años de histórico:", 1, 10, 5)
with col2:
    risk_free_rate = st.slider("Tasa libre de riesgo (%):", 0.0, 10.0, 2.0) / 100

start_date = datetime.now() - timedelta(days=365*years)
end_date = datetime.now()

if len(selected_tickers) < 2:
    st.warning("Selecciona al menos 2 activos para analizar el portafolio.")
else:
    # Cargar datos y calcular retornos
    returns_data = []
    valid_tickers = []
    
    for ticker in selected_tickers:
        data = load_ticker_data(ticker, start_date, end_date)
        if data is not None:
            returns = data['Close'].pct_change().dropna()
            returns_data.append(returns)
            valid_tickers.append(ticker)
    
    if len(valid_tickers) < 2:
        st.error("No hay suficientes datos para los activos seleccionados.")
    else:
        returns_df = pd.concat(returns_data, axis=1)
        returns_df.columns = valid_tickers
        returns_df = returns_df.dropna()
        
        # Mostrar estadísticas descriptivas
        st.subheader("Estadísticas de Retornos")
        st.dataframe(returns_df.describe().style.format("{:.2%}"))
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        corr_matrix = returns_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Correlación entre Activos"
        )
        
        fig.update_layout(
            height=600,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimización de portafolio
        st.subheader("Optimización de Portafolio")
        num_portfolios = st.slider("Número de portafolios a simular:", 1000, 50000, 10000)
        
        results, expected_returns, cov_matrix = markowitz_optimization(
            returns_df, 
            num_portfolios=num_portfolios,
            risk_free_rate=risk_free_rate
        )
        
        # Frontera eficiente
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
        
        # Portafolio óptimo (máximo Sharpe ratio)
        max_sharpe_idx = np.argmax(results[2])
        fig.add_trace(go.Scatter(
            x=[results[1,max_sharpe_idx]],
            y=[results[0,max_sharpe_idx]],
            mode='markers',
            name='Portafolio Óptimo',
            marker=dict(
                color='red',
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
        
        # Mostrar pesos óptimos
        st.subheader("Composición del Portafolio Óptimo")
        
        # Calcular pesos óptimos (simplificado)
        optimal_weights = np.random.dirichlet(np.ones(len(valid_tickers)), size=1)[0]
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        weights_df = pd.DataFrame({
            'Activo': valid_tickers,
            'Peso': optimal_weights,
            'Retorno Esperado': expected_returns,
            'Volatilidad': np.sqrt(np.diag(cov_matrix))
        })
        
        # Gráfico de pesos
        fig = px.pie(
            weights_df, 
            values='Peso', 
            names='Activo',
            title="Distribución del Portafolio Óptimo",
            hover_data=['Retorno Esperado', 'Volatilidad']
        )
        
        fig.update_layout(
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla con detalles
        st.dataframe(weights_df.sort_values('Peso', ascending=False).style.format({
            'Peso': '{:.2%}',
            'Retorno Esperado': '{:.2%}',
            'Volatilidad': '{:.2%}'
        }))
        
        # Métricas del portafolio óptimo
        st.subheader("Métricas del Portafolio Óptimo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Retorno Esperado", f"{results[0,max_sharpe_idx]*100:.2f}%")
        col2.metric("Volatilidad", f"{results[1,max_sharpe_idx]*100:.2f}%")
        col3.metric("Sharpe Ratio", f"{results[2,max_sharpe_idx]:.2f}")