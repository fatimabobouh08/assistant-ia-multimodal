from fastapi import FastAPI
from pydantic import BaseModel
from src.assistant.rag.chain import creer_chaine_rag

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Assistant RAG",
    description="API pour poser des questions sur des documents PDF",
    version="1.0.0"
)

# On charge le modèle IA et le retriever au démarrage pour ne pas ralentir chaque requête
print("Chargement du modèle...")
chaine_rag = creer_chaine_rag()

# On définit la structure des données attendues en entrée
class QuestionRequest(BaseModel):
    question: str

# On définit la structure des données renvoyées en sortie
class ReponseResponse(BaseModel):
    reponse: str

@app.get("/health")
def health_check():
    """Vérifie si l'API est en ligne."""
    return {"status": "ok"}

@app.post("/ask", response_model=ReponseResponse)
def poser_question(request: QuestionRequest):
    """Reçoit une question, interroge le RAG et renvoie la réponse."""
    print(f"Question reçue : {request.question}")
    
    # On invoque la chaîne RAG avec la question
    reponse = chaine_rag.invoke(request.question)
    
    # On renvoie la réponse formatée
    return {"reponse": reponse}

if __name__ == "__main__":
    import uvicorn
    # Lancement du serveur sur le port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)