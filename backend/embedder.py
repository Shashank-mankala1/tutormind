from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def adaptive_chunk_text(text, sentences_per_chunk=5):
    sentences = sent_tokenize(text, language="english")
    if len(sentences) < 2:
        return [text]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    num_chunks = max(1, len(sentences) // sentences_per_chunk)
    if num_chunks < 2:
        return [' '.join(sentences)] 
    
    clustering = AgglomerativeClustering(n_clusters=num_chunks)
    labels = clustering.fit_predict(embeddings)

    clustered = defaultdict(list)
    for sentence, label in zip(sentences, labels):
        clustered[label].append(sentence)

    chunks = [' '.join(clustered[k]) for k in sorted(clustered)]
    return chunks



def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

def create_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in chunks]
    metadata = [doc.metadata for doc in chunks]

    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
    vectorstore.save_local("vector_store/faiss_index")

    return vectorstore

