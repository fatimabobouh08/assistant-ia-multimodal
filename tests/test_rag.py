import os
import pytest
from langchain_community.document_loaders import PyPDFDirectoryLoader
from src.assistant.rag.retriever import obtenir_retriever

# Test 1 : Vérifier que les PDF sont présents et lisibles
def test_lecture_pdf():
    dossier_pdf = "data/pdfs"
    # On vérifie que le dossier existe
    assert os.path.exists(dossier_pdf), "Le dossier des PDF n'existe pas."
    
    # On essaie de lire les PDF
    loader = PyPDFDirectoryLoader(dossier_pdf)
    documents = loader.load()
    
    # On vérifie qu'on a bien lu au moins 1 page
    assert len(documents) > 0, "Aucun document n'a été lu."

# Test 2 : Vérifier que le retriever trouve bien des documents
def test_recherche_retriever():
    retriever = obtenir_retriever()
    question = "What is RAG?"
    
    # On demande au retriever de chercher
    resultats = retriever.invoke(question)
    
    # On vérifie qu'il a bien renvoyé 3 extraits (comme défini dans le code)
    assert len(resultats) == 3, "Le retriever n'a pas renvoyé le bon nombre d'extraits."
    
    # On vérifie que les extraits contiennent du texte
    for doc in resultats:
        assert len(doc.page_content) > 0, "Un extrait est vide."