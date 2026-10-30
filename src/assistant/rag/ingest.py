import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration des chemins
DOSSIER_PDF = "data/pdfs"
DOSSIER_INDEX = "data/faiss_index"

def main():
    print("🚀 Début de l'ingestion...")

    # 1. Chargement des documents
    print(f"📄 Lecture des PDF dans le dossier : {DOSSIER_PDF}")
    loader = PyPDFDirectoryLoader(DOSSIER_PDF)
    documents = loader.load()
    print(f"✅ {len(documents)} pages lues au total.")

    # 2. Découpage du texte (Chunking)
    # On utilise un overlap de 200 caractères pour éviter de couper une phrase 
    # en plein milieu et préserver le contexte entre les morceaux.
    print("✂️  Découpage du texte en morceaux...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Le texte a été découpé en {len(chunks)} morceaux (chunks).")

    # 3. Génération des embeddings
    # Utilisation d'un modèle local (HuggingFace) pour transformer le texte en vecteurs.
    # all-MiniLM-L6-v2 est léger, rapide et performant pour de la recherche sémantique.
    print("🧠 Création des embeddings (cela peut prendre un peu de temps la 1ère fois)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Indexation et sauvegarde locale avec FAISS
    print("💾 Création et sauvegarde de l'index FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DOSSIER_INDEX)
    
    print(f"🎉 Ingestion terminée avec succès ! L'index est sauvegardé dans : {DOSSIER_INDEX}")

if __name__ == "__main__":
    main()