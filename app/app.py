import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Segmentation Clients", layout="wide")

st.title("Application de Prédiction - Segmentation Client")
st.markdown("""
Cette application permet de déterminer à quel segment métier appartient un client 
en se basant sur ses habitudes de consommation et de paiement.
""")

@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_overall_model.joblib')
    pipeline = joblib.load(model_path)
    return pipeline

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("Le modèle est introuvable. Veuillez exécuter le script pipeline.py d'abord.")
    st.stop()

st.sidebar.header("Données d'entrée")

st.sidebar.subheader("Saisissez le profil du client :")

balance = st.sidebar.number_input("BALANCE", min_value=0.0, value=1000.0,step=1.0)
purchases = st.sidebar.number_input("PURCHASES", min_value=0.0, value=500.0,step=1.0)
oneoff = st.sidebar.number_input("ONEOFF_PURCHASES", min_value=0.0, value=200.0,step=1.0)
installments = st.sidebar.number_input("INSTALLMENTS_PURCHASES", min_value=0.0, value=300.0,step=1.0)
cash_advance = st.sidebar.number_input("CASH_ADVANCE", min_value=0.0, value=0.0,step=1.0)
credit_limit = st.sidebar.number_input("CREDIT_LIMIT", min_value=0.0, value=3000.0,step=1.0)
payments = st.sidebar.number_input("PAYMENTS", min_value=0.0, value=800.0,step=1.0)

nouveau_client_brut = pd.DataFrame({
    "BALANCE": [balance],
    "PURCHASES": [purchases],
    "ONEOFF_PURCHASES": [oneoff],
    "INSTALLMENTS_PURCHASES": [installments],
    "CASH_ADVANCE": [cash_advance],
    "CREDIT_LIMIT": [credit_limit],
    "PAYMENTS": [payments]
})

st.subheader("Aperçu du profil client brut saisi :")
st.dataframe(nouveau_client_brut)

if st.button("Prédire le segment"):
    with st.spinner("Interrogation du pipeline de Machine Learning..."):
        
        predictions = pipeline.predict(nouveau_client_brut)
        
        st.subheader("Résultats de la prédiction")
        
        segment_name = predictions[0]
        st.success(f"**Segment métier attribué : {segment_name}**")
    
