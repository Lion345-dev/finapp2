import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_ticker_data, get_sp500_tickers
from utils.calculations import calculate_returns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.title("🔮 Predicciones Financieras")
st.markdown("Modelos predictivos para precios y retornos de activos financieros.")

# Selección de ticker y parámetros
ticker = st.selectbox("Seleccionar activo:", get_sp500_tickers(), index=0, key='pred_ticker')
years = st.slider("Años de histórico:", 1, 10, 5)
start_date = datetime.now() - timedelta(days=365*years)
end_date = datetime.now()

# Cargar datos
data = load_ticker_data(ticker, start_date, end_date)

if data is not None:
    # Preparar datos
    data['Returns'] = calculate_returns(data['Close'])
    data = data.dropna()
    
    # Crear características
    lags = st.slider("Número de lags para el modelo:", 1, 20, 5)
    
    for lag in range(1, lags+1):
        data[f'Lag_{lag}'] = data['Returns'].shift(lag)
    
    data = data.dropna()
    X = data[[f'Lag_{lag}' for lag in range(1, lags+1)]]
    y = data['Returns']
    
    # Dividir datos
    test_size = st.slider("Tamaño del conjunto de prueba (%):", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Selección de modelo
    model_type = st.selectbox("Seleccionar modelo:", [
        "Regresión Lineal", 
        "Random Forest",
        "AR (Autoregresivo)"
    ])
    
    if model_type == "Regresión Lineal":
        model = LinearRegression()
    elif model_type == "Random Forest":
        n_estimators = st.slider("Número de árboles:", 10, 200, 100)
        max_depth = st.slider("Profundidad máxima:", 2, 20, 5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "AR (Autoregresivo)":
        # Modelo AR simple (promedio de lags)
        def ar_predict(X):
            return np.mean(X, axis=1)
    
    # Entrenar y predecir
    if model_type != "AR (Autoregresivo)":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        y_pred = ar_predict(X_test.values)
    
    # Métricas de evaluación
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    st.subheader("Evaluación del Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae*100:.4f}%")
    col2.metric("MSE", f"{mse*10000:.4f}%²")
    col3.metric("RMSE", f"{rmse*100:.4f}%")
    
    # Gráfico de predicciones vs real
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.iloc[-len(y_test):].index,
        y=y_test*100,
        name='Real',
        line=dict(color='#3498db')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.iloc[-len(y_test):].index,
        y=y_pred*100,
        name='Predicción',
        line=dict(color='#e74c3c', dash='dot')
    ))
    
    fig.update_layout(
        title="Predicciones vs Valores Reales",
        xaxis_title="Fecha",
        yaxis_title="Retorno Diario (%)",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Predicción hacia adelante
    st.subheader("Pronóstico Futuro")
    n_steps = st.slider("Número de días a predecir:", 1, 30, 5)
    
    if st.button("Generar Pronóstico"):
        last_values = data['Returns'].values[-lags:][::-1]
        predictions = []
        
        for _ in range(n_steps):
            if model_type != "AR (Autoregresivo)":
                pred = model.predict([last_values[:lags]])[0]
            else:
                pred = np.mean(last_values[:lags])
            
            predictions.append(pred)
            last_values = np.insert(last_values, 0, pred)[:lags]
        
        # Crear fechas futuras
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps+1)]
        
        # Gráfico de pronóstico
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index[-30:],
            y=data['Returns'].iloc[-30:]*100,
            name='Histórico',
            line=dict(color='#3498db')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=np.array(predictions)*100,
            name='Pronóstico',
            line=dict(color='#2ecc71', dash='dot')
        ))
        
        fig.update_layout(
            title=f"Pronóstico de Retornos para los próximos {n_steps} días",
            xaxis_title="Fecha",
            yaxis_title="Retorno Diario (%)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar predicciones en tabla
        forecast_df = pd.DataFrame({
            'Fecha': future_dates,
            'Retorno Predicho (%)': np.array(predictions)*100
        })
        
        st.dataframe(forecast_df.style.format({
            'Retorno Predicho (%)': '{:.4f}%'
        }))