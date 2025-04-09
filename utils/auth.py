import streamlit as st
import hashlib

def authenticate():
    """Sistema básico de autenticación"""
    
    # Verificar si ya está autenticado
    if st.session_state.get("authenticated", False):
        return True
    
    # Mostrar formulario de login
    with st.container():
        st.title("🔒 Acceso al Sistema Financiero")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/150", width=100)
        with col2:
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            
            if st.button("Ingresar"):
                # Validación simple (en producción usaría una base de datos)
                hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
                
                # Credenciales válidas (en producción esto vendría de una DB)
                valid_users = {
                    "admin": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # 'password'
                    "analyst": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92"  # '123456'
                }
                
                if username in valid_users and valid_users[username] == hashed_pwd:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas")
                    return False
    
    return False