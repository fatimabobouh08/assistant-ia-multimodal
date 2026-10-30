from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Chemin vers l'index que nous venons de créer
DOSSIER_INDEX = "data/faiss_index"

def obtenir_retriever():
    # 1. On recharge le même modèle d'embedding que lors de l'ingestion.
    # C'est crucial : la question et les documents doivent être "traduits" 
    # dans le même langage mathématique pour être comparés.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. On charge notre base de données FAISS depuis le disque dur
    # Le paramètre allow_dangerous_deserialization=True est nécessaire avec les 
    # versions récentes de LangChain pour charger un fichier .pkl (notre index.pkl).
    vectorstore = FAISS.load_local(
        DOSSIER_INDEX, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. On transforme la base de données en un "Retriever".
    # search_kwargs={"k": 3} signifie qu'à chaque question, 
    # l'outil ira chercher les 3 morceaux de texte les plus pertinents.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

if __name__ == "__main__":
    # Petit test pour vérifier que la recherche fonctionne
    print("🔍 Initialisation du retriever...")
    retriever = obtenir_retriever()
    
    question = "Qu'est-ce que le RAG ?"
    print(f"❓ Question test : {question}")
    
    # On demande au retriever de trouver les documents pertinents
    documents_trouves = retriever.invoke(question)
    
    print(f"\n✅ {len(documents_trouves)} extraits trouvés :")
    for i, doc in enumerate(documents_trouves):
        # On affiche les 200 premiers caractères de chaque extrait pour voir si c'est pertinent
        print(f"\n--- Extrait {i+1} ---")
        print(doc.page_content[:200] + "...")