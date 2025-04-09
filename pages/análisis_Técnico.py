import streamlit as st

# Configuración de la página - DEBE SER EL PRIMER COMANDO DE STREAMLIT
st.set_page_config(page_title="Análisis Técnico", layout="wide")

# Resto de imports
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from random import randint
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Intentar obtener API Key de diferentes fuentes
try:
    # Primero intentar desde Streamlit Secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Si no está en secrets, intentar desde .env
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configurar API Key con mejor manejo de errores
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        st.success("API de Gemini configurada correctamente")
    except Exception as e:
        st.error(f"Error al configurar Gemini: {str(e)}")
else:
    st.error("No se encontró la API Key de Gemini")

# Código de verificación (eliminar en producción)
if st.checkbox("Debug API Key"):
    if "GEMINI_API_KEY" in st.secrets:
        st.success("API Key encontrada en secretos de Streamlit")
    else:
        st.error("API Key no encontrada en secretos de Streamlit")

st.title("📈 Análisis Técnico Avanzado")
st.markdown("""
Herramienta profesional de análisis técnico con datos en tiempo real de Yahoo Finance.
Visualiza indicadores clave, patrones de velas y tendencias del mercado.
""")

# Inicializar estado de sesión
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search_input' not in st.session_state:
    st.session_state.search_input = ''

# Función para obtener tickers del S&P 500
@st.cache_data(ttl=86400)  # Cache por 24 horas
def get_sp500_tickers():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        tickers = df['Symbol'].tolist()
        additional_tickers = ['BTC-USD', 'ETH-USD', 'GC=F', 'CL=F', 'EURUSD=X']
        return sorted(tickers + additional_tickers)
    except Exception as e:
        st.error(f"Error al obtener tickers: {str(e)}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JNJ', 'JPM', 'V', 'BTC-USD']

# Función para cargar datos
@st.cache_data(ttl=3600, show_spinner="Obteniendo datos de mercado...")
def load_ticker_data(ticker, start_date, end_date):
    max_retries = 5  # Más reintentos
    
    for attempt in range(max_retries):
        try:
            # Delay progresivo
            time.sleep((attempt + 1) * 2)
            
            # Verificar si el ticker existe
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                raise ValueError(f"No se encontró información para {ticker}")
            
            # Intentar diferentes períodos si el específico falla
            try:
                # Primer intento con período específico
                data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False,
                    timeout=30,  # Timeout más largo
                    prepost=True,
                    interval='1d'  # Especificar intervalo diario
                )
                
                if data.empty:
                    # Intentar con períodos predefinidos
                    periods = ['1y', '2y', 'max']
                    for period in periods:
                        st.info(f"Intentando con período {period}...")
                        data = yf.download(
                            ticker,
                            period=period,
                            progress=False,
                            timeout=30,  # Timeout más largo
                            prepost=True,
                            interval='1d'  # Especificar intervalo diario
                        )
                        if not data.empty:
                            break
                
                if not data.empty:
                    # Limpiar y preparar datos
                    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                    data = data.ffill().bfill()
                    return data
                    
            except Exception as e:
                st.warning(f"Error en intento {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                continue
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error al cargar {ticker}: {str(e)}")
                return None
            continue
    
    return None

# Sidebar con configuración
with st.sidebar:
    st.header("Configuración")
    
    ticker_list = get_sp500_tickers()
    ticker = st.selectbox(
        "Seleccionar activo:",
        options=ticker_list,
        index=ticker_list.index('AAPL') if 'AAPL' in ticker_list else 0,
        help="Escribe para buscar un ticker específico"
    )
    
    # Campo de búsqueda manual
    search_input = st.text_input("Buscar ticker (ej: AAPL, BTC-USD)", value=st.session_state.search_input)
    if search_input != st.session_state.search_input:
        st.session_state.search_input = search_input
        ticker = search_input if search_input else ticker
    
    period = st.selectbox(
        "Período histórico:",
        options=['1m', '3m', '6m', '1y', '3y', '5y', '10y', 'Máximo'],
        index=3
    )
    
    st.markdown("---")
    st.markdown("**Indicadores técnicos**")
    show_ma = st.checkbox("Media Móvil", True)
    show_bb = st.checkbox("Bollinger Bands")
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_volume = st.checkbox("Volumen", True)
    
    if show_ma:
        ma_window = st.slider("Período Media Móvil:", 5, 200, 50)
    if show_bb:
        bb_window = st.slider("Período Bollinger Bands:", 5, 60, 20)
    if show_rsi:
        rsi_window = st.slider("Período RSI:", 5, 30, 14)

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
else:
    start_date = end_date - timedelta(days=365*20)

# Cargar datos
try:
    data = load_ticker_data(ticker, start_date, end_date)
    if data is not None:
        # Procesar datos
        st.success(f"Datos cargados exitosamente para {ticker}")
    else:
        st.error(f"No se pudieron cargar datos para {ticker}")
except Exception as e:
    st.error(f"Error inesperado: {str(e)}")

if data is not None and not data.empty:
    # Información básica del activo
    col1, col2, col3 = st.columns(3)
    
    # Format metrics with error handling
    try:
        last_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else last_close
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100

        with col1:
            st.metric(
                label=f"Precio {ticker}", 
                value=f"${last_close:,.2f}", 
                delta=f"{change:+,.2f} ({pct_change:+.2f}%)"
            )
    except (ValueError, TypeError, IndexError) as e:
        st.error(f"Error al procesar métricas de precio: {str(e)}")
        with col1:
            st.metric(
                label=f"Precio {ticker}", 
                value="N/A", 
                delta="N/A"
            )
    
    # Format volume metrics with error handling
    try:
        avg_volume = data['Volume'].mean()
        last_volume = data['Volume'].iloc[-1]
        # Check if avg_volume is valid before calculation
        if pd.notna(avg_volume) and avg_volume != 0:
            volume_change = (last_volume - avg_volume) / avg_volume * 100
        else:
            volume_change = 0
        
        with col2:
            st.metric(
                label="Volumen", 
                value=f"{int(last_volume):,}" if pd.notna(last_volume) else "N/A", 
                delta=f"{volume_change:.1f}% vs promedio" if pd.notna(volume_change) else "N/A"
            )
    except Exception as e:
        st.error(f"Error al procesar métricas de volumen: {str(e)}")
        with col2:
            st.metric(
                label="Volumen", 
                value="N/A", 
                delta="N/A"
            )
    
    # Calculate 52-week range with error handling
    try:
        # Ensure we have enough data and values are valid
        if len(data) >= 252:  # Approximately 252 trading days in a year
            week52_high = data['Close'].rolling(252).max().iloc[-1]
            week52_low = data['Close'].rolling(252).min().iloc[-1]
            
            with col3:
                if pd.notna(week52_high) and pd.notna(week52_low):
                    st.metric(
                        label="Rango 52 semanas", 
                        value=f"${week52_low:.2f} - ${week52_high:.2f}"
                    )
                else:
                    st.metric(
                        label="Rango 52 semanas",
                        value="N/A"
                    )
        else:
            # Use available range if we don't have enough data
            available_high = data['Close'].max()
            available_low = data['Close'].min()
            if pd.notna(available_high) and pd.notna(available_low):
                with col3:
                    st.metric(
                        label=f"Rango ({len(data)} días)",
                        value=f"${available_low:.2f} - ${available_high:.2f}"
                    )
            else:
                with col3:
                    st.metric(
                        label="Rango",
                        value="N/A"
                    )
    except Exception as e:
        st.error(f"Error al calcular rango de precios: {str(e)}")
        with col3:
            st.metric(
                label="Rango 52 semanas",
                value="N/A"
            )

    # Gráfico principal de velas
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Precios'))
    if show_ma:
        data[f'MA_{ma_window}'] = data['Close'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma_window}'], name=f'MA {ma_window}d'))
    if show_bb:
        data['BB_MA'] = data['Close'].rolling(window=bb_window).mean()
        data['BB_UP'] = data['BB_MA'] + 2 * data['Close'].rolling(window=bb_window).std()
        data['BB_DN'] = data['BB_MA'] - 2 * data['Close'].rolling(window=bb_window).std()
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_UP'], name='Banda Superior'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_DN'], name='Banda Inferior', fill='tonexty'))
    if show_volume:
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volumen', yaxis='y2'))
    fig.update_layout(
        title=f"{ticker} - Análisis Técnico",
        yaxis_title="Precio (USD)",
        template="plotly_dark",
        height=600,
        yaxis2=dict(title="Volumen", overlaying='y', side='right')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Indicadores adicionales (RSI, MACD, Análisis)
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Análisis"])
    with tab1:
        if show_rsi:
            st.subheader("Índice de Fuerza Relativa (RSI)")
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_window).mean()
            avg_loss = loss.rolling(window=rsi_window).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
            rsi_fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(rsi_fig, use_container_width=True)
    with tab2:
        if show_macd:
            st.subheader("MACD")
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Señal'))
            macd_fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(macd_fig, use_container_width=True)
    with tab3:
        st.subheader("Análisis Técnico Completo")
        st.write("Resumen de señales técnicas aquí...")  # Simplificado por brevedad

    # Sección adicional con información de la empresa
    stock = yf.Ticker(ticker)
    info = stock.info
    
    if info:
        if ticker not in st.session_state.history:
            st.session_state.history.append(ticker)
        
        st.markdown(f"""
            <div class="stock-card">
                <h2 style="color: #E5E9F0; margin:0;">{info.get('longName', ticker)}</h2>
                <p style="color: #D8DEE9; margin:0;">{info.get('sector', '')} | {info.get('industry', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📌 Descripción de la Empresa", expanded=True):
            if 'longBusinessSummary' in info:
                prompt = f"Traduce al español y resume en 3 párrafos máximo: {info['longBusinessSummary']}"
                try:
                    if GEMINI_API_KEY:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        st.write(response.text)
                    else:
                        st.warning("API Key de Gemini no configurada")
                except Exception as e:
                    st.warning(f"Error al procesar con Gemini: {str(e)}")
                    st.write(info['longBusinessSummary'])
            else:
                st.warning("Descripción no disponible")
        
        # Métricas clave
        st.markdown("<h3 style='color: #E5E9F0;'>📊 Métricas Clave</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precio Actual", f"${info.get('currentPrice', 'N/A'):,.2f}")
            st.metric("Mín. 52 Sem", f"${info.get('fiftyTwoWeekLow', 'N/A'):,.2f}")
        with col2:
            st.metric("Capitalización", f"${info.get('marketCap', 'N/A'):,.0f}")
            st.metric("Ratio P/E", f"{info.get('trailingPE', 'N/A'):,.1f}")
        with col3:
            st.metric("Máx. 52 Sem", f"${info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}")
            st.metric("Beta", f"{info.get('beta', 'N/A'):,.2f}")
        with col4:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            st.metric("Vol. Promedio", f"{info.get('averageVolume', 'N/A'):,}")
        
        # Noticias recientes
        st.markdown("<h3 style='color: #E5E9F0;'>📰 Noticias Recientes</h3>", unsafe_allow_html=True)
        news = stock.news
        for item in news[:5]:
            publish_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%d/%m/%Y %H:%M') if 'providerPublishTime' in item else ''
            st.markdown(f"""
                <div class="news-card">
                    <h4 style="color: #E5E9F0;">{item.get('title', 'Título no disponible')}</h4>
                    <p style="color: #D8DEE9;">{item.get('publisher', 'Fuente desconocida')} - {publish_time}</p>
                    <a href="{item.get('link', '#')}" target="_blank" style="color: #5E81AC;">Leer más →</a>
                </div>
                """, unsafe_allow_html=True)
        
        # Botón de favoritos
        if st.button("⭐ Agregar a Favoritos"):
            if ticker not in st.session_state.favorites:
                st.session_state.favorites.append(ticker)
                st.success(f"{ticker} agregado a favoritos!")
            else:
                st.warning(f"{ticker} ya está en tus favoritos")

else:
    st.error("No se pudieron cargar los datos para este activo.")

# Nota legal
st.markdown("---")
st.caption("""
📊 **Nota Legal:** Los datos son proporcionados por Yahoo Finance.  
💡 **Propósito Educativo:** Este análisis no constituye asesoramiento financiero.  
""")