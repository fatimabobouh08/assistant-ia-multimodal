# 🤖 Assistant IA Documentaire (RAG)

Un prototype d'assistant intelligent capable de répondre à des questions en se basant **uniquement** sur le contenu de documents PDF fournis. 

Ce projet implémente une architecture **RAG** (Retrieval-Augmented Generation) complète, avec une API et une interface utilisateur.

## 🎯 Fonctionnalités

- **Ingestion de documents** : Lecture de PDF, découpage intelligent en morceaux (chunks) et génération d'embeddings.
- **Recherche sémantique** : Utilisation de FAISS pour retrouver les passages les plus pertinents.
- **Génération de réponses** : Utilisation d'un LLM local (Ollama/Llama 3.2) pour synthétiser une réponse en français, sans hallucination.
- **API REST** : Exposée avec FastAPI (documentation interactive auto-générée).
- **Interface Utilisateur** : Application web interactive avec Streamlit.

## 🛠️ Stack Technique

- **Langage** : Python 3.12
- **IA / LLM** : LangChain, Ollama (Llama 3.2), HuggingFace (Sentence Transformers)
- **Base de données vectorielle** : FAISS
- **API** : FastAPI, Uvicorn
- **Interface** : Streamlit
- **Tests** : Pytest

## 📂 Architecture du projet

```text
assistant-ia-multimodal/
├── data/
│   ├── pdfs/                 # Dossier contenant les fichiers PDF à ingérer
│   └── faiss_index/          # Index FAISS généré (ne pas versionner)
├── src/
│   └── assistant/
│       ├── rag/
│       │   ├── ingest.py     # Script d'ingestion des PDF
│       │   ├── retriever.py  # Chargement de l'index et recherche
│       │   └── chain.py      # Chaîne RAG (Retriever + Prompt + LLM)
│       ├── api/
│       │   └── main.py       # Serveur FastAPI
│       └── ui/
│           └── app.py        # Interface Streamlit
├── tests/                    # Tests unitaires
├── requirements.txt          # Dépendances du projet
└── README.md