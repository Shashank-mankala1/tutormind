from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Chunk the extracted text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

# 2. Create and save FAISS vector store from chunks
def create_faiss_index(chunks, index_path="vector_store/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metadata = [{"source": f"chunk-{i}"} for i in range(len(chunks))]
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadata)   
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    return vectorstore  
