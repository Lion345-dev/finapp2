import streamlit as st
from PyQt5.QtCore import QSettings
import os
from dotenv import load_dotenv

def configure_page():
    """Configuración inicial de la página Streamlit"""
    st.set_page_config(
        page_title="FinApp - Análisis Financiero",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Cargar variables de entorno
    load_dotenv()

def set_dark_theme():
    """Aplicar tema oscuro a la aplicación"""
    # Configuración del tema oscuro
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        .css-18e3th9 {
            background-color: #121212;
        }
        .css-1d391kg {
            background-color: #1e1e1e;
        }
        .st-bw {
            background-color: #1e1e1e;
        }
        .st-at {
            background-color: #2e2e2e;
        }
        .st-ax {
            color: #ffffff;
        }
        .css-1aumxhk {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )