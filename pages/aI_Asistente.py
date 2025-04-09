import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as google_generativeai

# Configurar la API de Gemini
load_dotenv()

def configure_gemini():
    """Configurar la API de Gemini"""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        st.error("API key de Gemini no encontrada. Por favor configura la variable de entorno GEMINI_API_KEY.")
        return None
    
    try:
        google_generativeai.configure(api_key=api_key)
        st.success("API de Gemini configurada correctamente.")
    except Exception as e:
        st.error(f"Error al configurar la API de Gemini: {e}")
