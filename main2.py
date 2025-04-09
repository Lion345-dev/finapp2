import streamlit as st
from utils.auth import authenticate
from utils.config import configure_page, set_dark_theme
import os


# Configuración inicial de la página
configure_page()

# Aplicar tema oscuro
set_dark_theme()

# Sistema de autenticación básico
if not authenticate():
    st.stop()

# Barra lateral para navegación y configuración
st.sidebar.title("🔍 Navegación")
page = st.sidebar.radio("Ir a:", [
    "📊 Dashboard", 
    "📈 Análisis Técnico", 
    "📉 Medición de Riesgo", 
    "🔮 Predicciones", 
    "📑 Optimización de Portafolio",
    "💡 Asistente AI"
])

# Mostrar la página seleccionada
if page == "📊 Dashboard Principal":
    from pages import dashboard
elif page == "📈 Análisis Técnico":
    from pages import análisis_Técnico
elif page == "📉 Medición de Riesgo":
    from pages import riesgo
elif page == "🔮 Predicciones":
    from pages import predicciones
elif page == "📑 Optimización de Portafolio":
    from pages import portafolio
elif page == "💡 Asistente AI":
    from pages import aI_Asistente

# Footer de la aplicación
st.sidebar.markdown("---")
st.sidebar.markdown("""git init

**Configuración avanzada**  
🔧 Ajustes de visualización  
⚙️ Preferencias de cálculo  
""")

# Mostrar versión y estado
st.sidebar.markdown("---")
st.sidebar.caption(f"Versión 1.0 | Entorno: {os.getenv('ENVIRONMENT', 'development')}")