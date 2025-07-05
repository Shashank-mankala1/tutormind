from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from backend.model_router import query_model
from collections import Counter

def load_faiss_index(index_path="vector_store/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_pipeline(model_choice, provider):
    if provider != "local":
        return None 
    task = "text2text-generation"
    if any(key in model_choice.lower() for key in ["llama", "mistral", "phi", "tinyllama"]):
        task = "text-generation"
    return pipeline(
        task,
        model=model_choice,
        tokenizer=model_choice,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.7
    )

def rerank_documents(query, documents, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_scores = []

    for doc in documents:
        doc_embedding = model.encode(doc.page_content, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, doc_embedding).item()
        doc_scores.append((score, doc))

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in doc_scores[:top_k]]

def build_qa_chain(vectorstore, model_choice="mistralai/Mistral-7B-Instruct-v0.1", mode="Conceptual (Generative)", provider="local"):
    if mode == "Factual (Extractive)":
        qa_model = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        def extractive_qa(query):
            selected_files = st.session_state.get("selected_files", None)
            docs = vectorstore.similarity_search(query, k=20)
            if selected_files:
                docs = [doc for doc in docs if doc.metadata.get("source") in selected_files]
            docs = docs[:10]

            query_embedding = sentence_model.encode(query, convert_to_tensor=True)
            doc_scores = [(util.cos_sim(query_embedding, sentence_model.encode(doc.page_content, convert_to_tensor=True)).item(), doc) for doc in docs]
            doc_scores.sort(reverse=True)
            top_docs = [doc for score, doc in doc_scores[:3]]

            file_counts = Counter(doc.metadata.get("source", "Unknown") for doc in top_docs)
            most_common_file, _ = file_counts.most_common(1)[0]
            top_docs = [doc for doc in top_docs if doc.metadata.get("source") == most_common_file]

            context = "\n".join([doc.page_content for doc in top_docs])

            result = qa_model(question=query, context=context)
            answer = result['answer']
            confidence = result.get('score', 0)

            return {
                "result": f"{answer} (Confidence: {round(confidence * 100, 1)}%)",
                "source_documents": top_docs
            }

        return extractive_qa

    else:
        def generative_qa(query):
            selected_files = st.session_state.get("selected_files", None)
            docs = vectorstore.similarity_search(query, k=20)
            if selected_files:
                docs = [doc for doc in docs if doc.metadata.get("source") in selected_files]
            docs = docs[:10]

            top_docs = rerank_documents(query, docs, top_k=5)

            file_counts = Counter(doc.metadata.get("source", "Unknown") for doc in top_docs)
            most_common_file, _ = file_counts.most_common(1)[0]
            top_docs = [doc for doc in top_docs if doc.metadata.get("source") == most_common_file]

            context = "\n\n".join([doc.page_content for doc in top_docs])

            prompt = f"""You are a helpful assistant. Use ONLY the following context to answer the user's question. \
If the answer is not in the document, respond with \"Not found in the document.\"

<context>
{context}
</context>

<question>
{query}
</question>

<answer>
"""
            answer = query_model(prompt, provider=provider, model=model_choice)

            return {
                "result": answer,
                "source_documents": top_docs
            }

        return generative_qa