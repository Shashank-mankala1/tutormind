from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

def load_faiss_index(index_path="vector_store/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_pipeline(model_choice):
    return pipeline("text2text-generation", model=model_choice, tokenizer=model_choice, max_length=256)


def rerank_documents(query, documents, top_k=2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_scores = []

    for doc in documents:
        doc_embedding = model.encode(doc.page_content, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, doc_embedding).item()
        doc_scores.append((score, doc))

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in doc_scores[:top_k]]

def build_qa_chain(vectorstore, model_choice="google/flan-t5-base", mode="Conceptual (Generative)"):
    if mode == "Factual (Extractive)":
        qa_model = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        def extractive_qa(query):
            docs = vectorstore.similarity_search(query, k=10)

            query_embedding = sentence_model.encode(query, convert_to_tensor=True)
            doc_scores = [(util.cos_sim(query_embedding, sentence_model.encode(doc.page_content, convert_to_tensor=True)).item(), doc) for doc in docs]
            doc_scores.sort(reverse=True)

            top_docs = [doc for score, doc in doc_scores[:3]]
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
        qa_pipe = load_pipeline(model_choice)
        llm = HuggingFacePipeline(pipeline=qa_pipe)

        prompt = PromptTemplate.from_template("""
        Use the following course syllabus context to answer the question.
        If the answer is not explicitly mentioned, say "Not found in the document."

        Context:
        {context}

        Question: {question}

        Answer:
        """)

        chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        def generative_qa(query):
            docs = vectorstore.similarity_search(query, k=10)
            top_docs = rerank_documents(query, docs, top_k=2)
            return {
                "result": chain.invoke({"context": top_docs, "question": query}),
                "source_documents": top_docs
            }

        return generative_qa
