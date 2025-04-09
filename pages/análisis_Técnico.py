import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from random import randint

# Configuración de la página
st.set_page_config(page_title="Análisis Técnico", layout="wide")
st.title("📈 Análisis Técnico Avanzado")
st.markdown("""
Herramienta profesional de análisis técnico con datos en tiempo real de Yahoo Finance.
Visualiza indicadores clave, patrones de velas y tendencias del mercado.
""")

# Función para obtener tickers del S&P 500 (corregida)
@st.cache_data(ttl=86400)  # Cache por 24 horas
def get_sp500_tickers():
    """Obtiene la lista actualizada de tickers del S&P 500 con formato correcto para yfinance"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        # Convertir tickers al formato de Yahoo Finance (BRK.B -> BRK-B)
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        tickers = df['Symbol'].tolist()
        
        # Añadir algunos tickers comunes adicionales
        additional_tickers = ['BTC-USD', 'ETH-USD', 'GC=F', 'CL=F', 'EURUSD=X']
        return sorted(tickers + additional_tickers)
    except Exception as e:
        st.error(f"Error al obtener tickers: {str(e)}")
        # Lista de respaldo con tickers populares
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JNJ', 'JPM', 'V', 'BTC-USD']

# Función para cargar datos (versión robusta)
@st.cache_data(ttl=3600, show_spinner="Obteniendo datos de mercado...")
def load_ticker_data(ticker, start_date, end_date):
    """Carga datos históricos de Yahoo Finance con manejo de errores mejorado"""
    try:
        # Pequeño retardo aleatorio para evitar bloqueos
        time.sleep(randint(1, 3))
        
        # Convertir fechas a string
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')  # Incluir día final
        
        # Descargar datos
        data = yf.download(
            ticker,
            start=start_str,
            end=end_str,
            progress=False,
            timeout=10
        )
        
        # Si no hay datos, intentar con el máximo histórico
        if data.empty:
            st.warning(f"Intentando descargar el historico completo para {ticker}...")
            data = yf.download(ticker, period="max", progress=False)
            if not data.empty:
                data = data.loc[start_str:end_str]
        
        # Verificar si se obtuvieron datos
        if data.empty:
            st.error(f"No se encontraron datos para {ticker} en el rango especificado")
            return None
        
        # Limpiar y completar datos
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().bfill()
        
        return data
    
    except Exception as e:
        st.error(f"Error al cargar datos para {ticker}: {str(e)}")
        return None

# Sidebar con configuración
with st.sidebar:
    st.header("Configuración")
    
    # Selector de ticker con búsqueda
    ticker_list = get_sp500_tickers()
    ticker = st.selectbox(
        "Seleccionar activo:",
        options=ticker_list,
        index=ticker_list.index('AAPL') if 'AAPL' in ticker_list else 0,
        help="Escribe para buscar un ticker específico"
    )
    
    # Selector de período
    period = st.selectbox(
        "Período histórico:",
        options=['1m', '3m', '6m', '1y', '3y', '5y', '10y', 'Máximo'],
        index=3,
        help="Selecciona el rango temporal para el análisis"
    )
    
    st.markdown("---")
    st.markdown("**Indicadores técnicos**")
    
    # Opciones de indicadores
    show_ma = st.checkbox("Media Móvil", True)
    show_bb = st.checkbox("Bollinger Bands")
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_volume = st.checkbox("Volumen", True)
    
    # Parámetros avanzados
    if show_ma:
        ma_window = st.slider("Período Media Móvil:", 5, 200, 50)
    
    if show_bb:
        bb_window = st.slider("Período Bollinger Bands:", 5, 60, 20)
    
    if show_rsi:
        rsi_window = st.slider("Período RSI:", 5, 30, 14)

# Determinar rango de fechas según período seleccionado
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

if data is not None and not data.empty:
    # Mostrar información básica del activo
    col1, col2, col3 = st.columns(3)
    
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    
    with col1:
        st.metric(
            label=f"Precio {ticker}",
            value=f"${last_close:.2f}",
            delta=f"{change:.2f} ({pct_change:.2f}%)",
            delta_color="inverse" if change < 0 else "normal"
        )
    
    with col2:
        avg_volume = data['Volume'].mean()
        last_volume = data['Volume'].iloc[-1]
        volume_change = (last_volume - avg_volume) / avg_volume * 100
        st.metric(
            label="Volumen",
            value=f"{last_volume:,.0f}",
            delta=f"{volume_change:.1f}% vs promedio",
            help="Comparación con el volumen promedio del período seleccionado"
        )
    
    with col3:
        week52_high = data['Close'].rolling(252).max().iloc[-1]
        week52_low = data['Close'].rolling(252).min().iloc[-1]
        st.metric(
            label="Rango 52 semanas",
            value=f"${week52_low:.2f} - ${week52_high:.2f}",
            delta=f"{(last_close - week52_low)/(week52_high - week52_low)*100:.1f}% del rango",
            help="Posición actual dentro del rango anual"
        )

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
        increasing_line_color='#2ECC71',  # Verde
        decreasing_line_color='#E74C3C'   # Rojo
    ))

    # Media móvil
    if show_ma:
        data[f'MA_{ma_window}'] = data['Close'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[f'MA_{ma_window}'],
            name=f'MA {ma_window}d',
            line=dict(color='#F39C12', width=2),  # Naranja
            visible=True
        ))

    # Bollinger Bands
    if show_bb:
        data['BB_MA'] = data['Close'].rolling(window=bb_window).mean()
        data['BB_UP'] = data['BB_MA'] + 2 * data['Close'].rolling(window=bb_window).std()
        data['BB_DN'] = data['BB_MA'] - 2 * data['Close'].rolling(window=bb_window).std()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_UP'],
            name='Banda Superior',
            line=dict(color='rgba(52, 152, 219, 0.5)'),
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_DN'],
            name='Banda Inferior',
            line=dict(color='rgba(52, 152, 219, 0.5)'),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.1)',
            hoverinfo='skip'
        ))

    # Volumen
    if show_volume:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volumen',
            yaxis='y2',
            marker_color=np.where(data['Close'].diff() >= 0, '#2ECC71', '#E74C3C'),
            opacity=0.5
        ))

    # Configuración del layout del gráfico principal
    fig.update_layout(
        title=f"{ticker} - Análisis Técnico ({start_date.strftime('%Y-%m-%d')} al {end_date.strftime('%Y-%m-%d')})",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis2=dict(
            title="Volumen",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Indicadores adicionales en pestañas
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Análisis"])

    with tab1:
        if show_rsi:
            st.subheader("Índice de Fuerza Relativa (RSI)")
            
            # Calcular RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_window).mean()
            avg_loss = loss.rolling(window=rsi_window).mean()
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Gráfico RSI
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='#9B59B6', width=2)
            ))
            
            # Zonas de sobrecompra/sobreventa
            rsi_fig.add_hrect(
                y0=70, y1=100,
                fillcolor="rgba(231, 76, 60, 0.1)",
                line_width=0,
                annotation_text="Sobrecompra",
                annotation_position="top right"
            )
            
            rsi_fig.add_hrect(
                y0=0, y1=30,
                fillcolor="rgba(46, 204, 113, 0.1)",
                line_width=0,
                annotation_text="Sobreventa",
                annotation_position="bottom right"
            )
            
            rsi_fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="rgba(231, 76, 60, 0.7)",
                annotation_text="70",
                annotation_position="top right"
            )
            
            rsi_fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="rgba(46, 204, 113, 0.7)",
                annotation_text="30",
                annotation_position="bottom right"
            )
            
            rsi_fig.update_layout(
                height=300,
                template="plotly_dark",
                showlegend=False,
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(rsi_fig, use_container_width=True)
            
            # Interpretación RSI
            last_rsi = data['RSI'].iloc[-1]
            
            if last_rsi > 70:
                st.warning("""
                **RSI indica SOBRECOMPRA (por encima de 70)**  
                El activo podría estar sobrevalorado y podría producirse una corrección a la baja.
                """)
            elif last_rsi < 30:
                st.success("""
                **RSI indica SOBREVENTA (por debajo de 30)**  
                El activo podría estar infravalorado y podría producirse un rebote al alza.
                """)
            else:
                st.info("""
                **RSI en rango neutral (30-70)**  
                El activo no muestra señales fuertes de sobrecompra o sobreventa.
                """)

    with tab2:
        if show_macd:
            st.subheader("MACD (Convergencia/Divergencia de Medias Móviles)")
            
            # Calcular MACD
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['Histogram'] = data['MACD'] - data['Signal']
            
            # Gráfico MACD
            macd_fig = go.Figure()
            
            # Histograma
            macd_fig.add_trace(go.Bar(
                x=data.index,
                y=data['Histogram'],
                name='Histograma',
                marker_color=np.where(data['Histogram'] < 0, '#E74C3C', '#2ECC71'),
                opacity=0.6
            ))
            
            # Línea MACD
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='#3498DB', width=2)
            ))
            
            # Línea de señal
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Signal'],
                name='Señal',
                line=dict(color='#F39C12', width=2)
            ))
            
            macd_fig.update_layout(
                height=300,
                template="plotly_dark",
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(macd_fig, use_container_width=True)
            
            # Interpretación MACD
            current_macd = data['MACD'].iloc[-1]
            current_signal = data['Signal'].iloc[-1]
            prev_macd = data['MACD'].iloc[-2]
            prev_signal = data['Signal'].iloc[-2]
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                st.success("""
                **Señal de COMPRA: MACD cruzó al alza la línea de señal**  
                Indica posible inicio de tendencia alcista.
                """)
            elif current_macd < current_signal and prev_macd >= prev_signal:
                st.error("""
                **Señal de VENTA: MACD cruzó a la baja la línea de señal**  
                Indica posible inicio de tendencia bajista.
                """)
            elif current_macd > 0 and current_macd > current_signal:
                st.info("""
                **Tendencia alcista en desarrollo**  
                MACD positivo y por encima de la línea de señal.
                """)
            elif current_macd < 0 and current_macd < current_signal:
                st.info("""
                **Tendencia bajista en desarrollo**  
                MACD negativo y por debajo de la línea de señal.
                """)
            else:
                st.info("""
                **Mercado en consolidación**  
                No hay señales claras de tendencia.
                """)

    with tab3:
        st.subheader("Análisis Técnico Completo")
        
        # Resumen de señales
        st.markdown("### Resumen de Señales Técnicas")
        
        # Inicializar contadores de señales
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        # Evaluar RSI
        if show_rsi:
            last_rsi = data['RSI'].iloc[-1]
            if last_rsi < 30:
                buy_signals += 1
                st.success("✅ RSI en zona de SOBREVENTA (potencial compra)")
            elif last_rsi > 70:
                sell_signals += 1
                st.error("❌ RSI en zona de SOBRECOMPRA (potencial venta)")
            else:
                neutral_signals += 1
                st.info("➖ RSI en zona neutral")
        
        # Evaluar MACD
        if show_macd:
            current_macd = data['MACD'].iloc[-1]
            current_signal = data['Signal'].iloc[-1]
            prev_macd = data['MACD'].iloc[-2]
            prev_signal = data['Signal'].iloc[-2]
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                buy_signals += 1
                st.success("✅ MACD cruzó al alza (señal de compra)")
            elif current_macd < current_signal and prev_macd >= prev_signal:
                sell_signals += 1
                st.error("❌ MACD cruzó a la baja (señal de venta)")
            else:
                neutral_signals += 1
                st.info("➖ MACD sin cruces recientes")
        
        # Evaluar Media Móvil
        if show_ma:
            current_close = data['Close'].iloc[-1]
            current_ma = data[f'MA_{ma_window}'].iloc[-1]
            
            if current_close > current_ma:
                buy_signals += 0.5  # Media ponderación
                st.success(f"✅ Precio sobre MA{ma_window} (sesgo alcista)")
            else:
                sell_signals += 0.5
                st.error(f"❌ Precio bajo MA{ma_window} (sesgo bajista)")
        
        # Recomendación consolidada
        st.markdown("### Recomendación Consolidada")
        
        if buy_signals > sell_signals and buy_signals >= 2:
            st.success("""
            **📈 FUERTE SEÑAL DE COMPRA**  
            Múltiples indicadores sugieren oportunidad de entrada alcista.
            """)
        elif buy_signals > sell_signals:
            st.success("""
            **📈 Señal de compra**  
            Algunos indicadores sugieren oportunidad alcista.
            """)
        elif sell_signals > buy_signals and sell_signals >= 2:
            st.error("""
            **📉 FUERTE SEÑAL DE VENTA**  
            Múltiples indicadores sugieren riesgo bajista.
            """)
        elif sell_signals > buy_signals:
            st.error("""
            **📉 Señal de venta**  
            Algunos indicadores sugieren riesgo bajista.
            """)
        else:
            st.info("""
            **⚖️ Mercado neutral**  
            Los indicadores no muestran señales claras.
            """)
        
        # Datos estadísticos
        st.markdown("### Estadísticas Clave")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Volatilidad (ATR)", 
                     f"{(data['High'] - data['Low']).mean():.2f}",
                     help="Rango promedio verdadero")
        
        with col2:
            st.metric("Correlación RSI/Precio",
                     f"{data['RSI'].corr(data['Close']):.2f}" if 'RSI' in data else "N/A",
                     help="Correlación de 30 días")
        
        with col3:
            st.metric("Días consecutivos",
                     f"{len(data)} días",
                     help="Período de análisis actual")

# Mensaje si no hay datos
elif data is None or data.empty:
    st.error("""
    **No se pudieron cargar los datos para este activo.**  
    Posibles causas:  
    - El ticker no existe en Yahoo Finance  
    - Problemas de conexión a internet  
    - El activo no tiene datos históricos  
    - El rango de fechas seleccionado no tiene datos  
    
    **Solución:**  
    - Verifica que el ticker sea correcto (ej: AAPL, MSFT, BTC-USD)  
    - Prueba con un rango de fechas diferente  
    - Intenta recargar la página  
    """)

# Nota legal al pie
st.markdown("---")
st.caption("""
📊 **Nota Legal:** Los datos son proporcionados por Yahoo Finance.  
💡 **Propósito Educativo:** Este análisis no constituye asesoramiento financiero.  
⚠️ **Advertencia de Riesgo:** El trading conlleva riesgos de pérdida de capital.  
🔄 **Actualización:** Datos con retraso de 15-20 minutos para mercados estadounidenses.  
""")