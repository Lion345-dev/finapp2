import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Análisis Técnico", layout="wide")
st.title("📈 Análisis Técnico Avanzado")
st.markdown("""
Herramienta profesional de análisis técnico con datos en tiempo real de Yahoo Finance.
Visualiza indicadores clave, patrones de velas y tendencias del mercado.
""")

# Función para obtener tickers del S&P 500
@st.cache_data
def get_sp500_tickers():
    """Obtiene la lista actualizada de tickers del S&P 500"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Formatear tickers para yfinance (reemplazar . por -)
        tickers = [t.replace('.', '-') for t in tickers]
        return sorted(tickers)
    except Exception as e:
        st.error(f"Error al obtener tickers: {str(e)}")
        # Lista de respaldo
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']

# Función para cargar datos
@st.cache_data
def load_ticker_data(ticker, start_date, end_date):
    """Carga datos históricos de Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No se encontraron datos para {ticker}")
            return None
        
        # Calcular retornos diarios
        data['Daily Return'] = data['Close'].pct_change()
        
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

# Sidebar con configuración
with st.sidebar:
    st.header("Configuración")
    ticker = st.selectbox("Seleccionar activo:", get_sp500_tickers(), index=0)
    period = st.selectbox("Período histórico:", ['1m', '3m', '6m', '1y', '3y', '5y', '10y', 'Máximo'], index=3)
    
    st.markdown("---")
    st.markdown("**Indicadores técnicos**")
    show_ma = st.checkbox("Media Móvil", True)
    show_bb = st.checkbox("Bollinger Bands")
    show_rsi = st.checkbox("RSI")
    show_macd = st.checkbox("MACD")
    show_volume = st.checkbox("Volumen", True)
    
    if show_ma:
        ma_window = st.slider("Período Media Móvil:", 5, 200, 50)
    
    if show_bb:
        bb_window = st.slider("Período Bollinger Bands:", 5, 60, 20)

# Determinar rango de fechas
end_date = datetime.now()
if period == '1m':
    start_date = end_date - timedelta(days=30)
elif period == '3m':
    start_date = end_date - timedelta(days=90)
elif period == '6m':
    start_date = end_date - timedelta(days=180)
elif period == '1y':
    start_date = end_date - timedelta(days=365)
elif period == '3y':
    start_date = end_date - timedelta(days=365*3)
elif period == '5y':
    start_date = end_date - timedelta(days=365*5)
elif period == '10y':
    start_date = end_date - timedelta(days=365*10)
else:  # Máximo
    start_date = end_date - timedelta(days=365*20)  # Yahoo Finance suele tener ~20 años de datos

# Cargar datos
data = load_ticker_data(ticker, start_date, end_date)

if data is not None:
    # Mostrar información básica del activo
    col1, col2, col3 = st.columns(3)
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    
    with col1:
        st.metric(f"Precio {ticker}", f"${last_close:.2f}", 
                 f"{change:.2f} ({pct_change:.2f}%)", 
                 "inverse" if change < 0 else "normal")
    
    with col2:
        st.metric("Volumen promedio", f"{data['Volume'].mean():,.0f}", 
                 f"Último: {data['Volume'].iloc[-1]:,.0f}")
    
    with col3:
        st.metric("Rango 52 semanas", 
                 f"${data['Close'].rolling(252).min().iloc[-1]:.2f} - ${data['Close'].rolling(252).max().iloc[-1]:.2f}")

    # Gráfico principal de velas
    fig = go.Figure()

    # Velas japonesas
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Precios',
        increasing_line_color='#2ECC71',
        decreasing_line_color='#E74C3C'
    ))

    # Media móvil
    if show_ma:
        data[f'MA_{ma_window}'] = data['Close'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[f'MA_{ma_window}'],
            name=f'MA {ma_window}d',
            line=dict(color='#F39C12', width=2)
        ))

    # Bollinger Bands
    if show_bb:
        data['BB_MA'] = data['Close'].rolling(window=bb_window).mean()
        data['BB_UP'] = data['BB_MA'] + 2 * data['Close'].rolling(window=bb_window).std()
        data['BB_DN'] = data['BB_MA'] - 2 * data['Close'].rolling(window=bb_window).std()
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_UP'],
            name='Banda Superior',
            line=dict(color='rgba(255,255,255,0.2)')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_DN'],
            name='Banda Inferior',
            line=dict(color='rgba(255,255,255,0.2)'),
            fill='tonexty',
            fillcolor='rgba(50,50,50,0.1)'
        ))

    # Volumen
    if show_volume:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volumen',
            yaxis='y2',
            marker_color='rgba(100, 150, 255, 0.4)'
        ))

    # Configuración del layout
    fig.update_layout(
        title=f"{ticker} - Análisis Técnico",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis2=dict(
            title="Volumen",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Indicadores adicionales en pestañas
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Análisis"])

    with tab1:
        if show_rsi:
            st.subheader("Índice de Fuerza Relativa (RSI)")
            window_rsi = st.slider("Período RSI:", 5, 30, 14)
            
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window_rsi).mean()
            avg_loss = loss.rolling(window=window_rsi).mean()
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='#9B59B6', width=2)
            ))
            
            rsi_fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1)
            rsi_fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1)
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            
            rsi_fig.update_layout(
                height=300,
                template="plotly_dark",
                showlegend=False
            )
            
            st.plotly_chart(rsi_fig, use_container_width=True)
            
            # Interpretación RSI
            last_rsi = data['RSI'].iloc[-1]
            if last_rsi > 70:
                st.warning("RSI indica SOBRECOMPRA (por encima de 70)")
            elif last_rsi < 30:
                st.success("RSI indica SOBREVENTA (por debajo de 30)")
            else:
                st.info("RSI en rango neutral (30-70)")

    with tab2:
        if show_macd:
            st.subheader("MACD (Convergencia/Divergencia de Medias Móviles)")
            
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['Histogram'] = data['MACD'] - data['Signal']
            
            macd_fig = go.Figure()
            
            macd_fig.add_trace(go.Bar(
                x=data.index,
                y=data['Histogram'],
                name='Histograma',
                marker_color=np.where(data['Histogram'] < 0, '#E74C3C', '#2ECC71')
            ))
            
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='#3498DB', width=2)
            ))
            
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Signal'],
                name='Señal',
                line=dict(color='#F39C12', width=2)
            ))
            
            macd_fig.update_layout(
                height=300,
                template="plotly_dark",
                hovermode="x unified"
            )
            
            st.plotly_chart(macd_fig, use_container_width=True)
            
            # Interpretación MACD
            if data['MACD'].iloc[-1] > data['Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['Signal'].iloc[-2]:
                st.success("Señal de COMPRA: MACD cruzó al alza la línea de señal")
            elif data['MACD'].iloc[-1] < data['Signal'].iloc[-1] and data['MACD'].iloc[-2] >= data['Signal'].iloc[-2]:
                st.error("Señal de VENTA: MACD cruzó a la baja la línea de señal")

    with tab3:
        st.subheader("Análisis Técnico Avanzado")
        
        # Patrones de velas
        st.markdown("**Patrones de velas detectados**")
        # Aquí podrías implementar detección de patrones (Doji, Martillo, etc.)
        
        # Soporte y resistencia
        st.markdown("**Niveles clave**")
        # Aquí podrías implementar detección automática de soportes/resistencias
        
        # Recomendación basada en indicadores
        st.markdown("**Recomendación técnica**")
        # Lógica simple basada en múltiples indicadores
        buy_signals = 0
        sell_signals = 0
        
        if show_rsi:
            if data['RSI'].iloc[-1] < 30:
                buy_signals += 1
            elif data['RSI'].iloc[-1] > 70:
                sell_signals += 1
        
        if show_macd:
            if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
                buy_signals += 1
            else:
                sell_signals += 1
        
        if buy_signals > sell_signals:
            st.success("✅ Tendencia alcista - Considerar posiciones largas")
        elif sell_signals > buy_signals:
            st.error("🔻 Tendencia bajista - Considerar posiciones cortas")
        else:
            st.info("⚖️ Mercado neutral - Esperar confirmación")

# Nota al pie
st.markdown("---")
st.caption("""
📊 Datos proporcionados por Yahoo Finance. Análisis con fines educativos solamente.  
El rendimiento pasado no es indicador de resultados futuros. Invertir conlleva riesgos.
""")