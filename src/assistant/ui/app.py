import sys
import os
# Ajoute le dossier racine au chemin de recherche de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import streamlit as st
from src.assistant.rag.chain import creer_chaine_rag

# 1. Configuration de la page (titre de l'onglet, icône, mise en page)
st.set_page_config(
    page_title="Assistant IA - Analyse de documents",
    page_icon="🤖",
    layout="centered"
)

# 2. Titre et description
st.title("🤖 Assistant IA Documentaire")
st.markdown("""
Posez une question sur le contenu des articles scientifiques (RAG, CLIP, Sécurité du RAG).
L'assistant utilisera l'IA locale pour vous répondre en se basant uniquement sur les textes fournis.
""")

# 3. Initialisation de la chaîne RAG (on le fait une seule fois pour ne pas ralentir)
# Le décorateur @st.cache_resource permet de garder le modèle en mémoire
@st.cache_resource
def charger_chaine():
    return creer_chaine_rag()

try:
    with st.spinner("Chargement du modèle d'IA en cours..."):
        chaine_rag = charger_chaine()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# 4. Création de la zone de chat
# On initialise l'historique des messages s'il n'existe pas encore
if "messages" not in st.session_state:
    st.session_state.messages = []

# On affiche les messages précédents de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Barre de saisie de l'utilisateur
if prompt := st.chat_input("Posez votre question ici (ex: What is RAG?)..."):
    # Ajout de la question de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("L'IA réfléchit..."):
            reponse = chaine_rag.invoke(prompt)
            st.markdown(reponse)
            
    # Ajout de la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse})