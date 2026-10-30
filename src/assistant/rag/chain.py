from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.assistant.rag.retriever import obtenir_retriever

def formater_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def creer_chaine_rag():
    llm = ChatOllama(model="llama3.2", temperature=0)

    retriever = obtenir_retriever()

    # On améliore le prompt pour guider l'IA
    template = """Tu es un assistant de recherche très précis.
Voici des extraits de documents pour t'aider à répondre :
---
{context}
---

En te basant UNIQUEMENT sur ces extraits, réponds à la question suivante de façon claire et concise. 
Si la réponse n'est pas dans les extraits, dis "Je ne sais pas".

Question : {question}
Réponse en français :"""
    
    prompt = PromptTemplate.from_template(template)

    chaine = (
        {"context": retriever | formater_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chaine

if __name__ == "__main__":
    print("🤖 Initialisation de l'assistant IA local (Ollama)...")
    chaine_rag = creer_chaine_rag()
    
    question = "What is RAG and how does it work?"
    print(f"\n❓ Question : {question}")
    print("⏳ L'IA réfléchit (ça peut prendre 10-30 secondes)...")
    
    reponse = chaine_rag.invoke(question)
    
    print(f"\n💡 Réponse de l'IA :\n{reponse}")