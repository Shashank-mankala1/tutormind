import streamlit as st
from backend.loader import load_any_file
from backend.qa_chain import load_faiss_index, build_qa_chain
from backend.embedder import adaptive_chunk_text, create_faiss_index
from backend.firebase_db import save_qa, get_history, clear_history, submit_app_feedback
from backend.firebase_auth import register_user, validate_user
from backend.model_router import query_model
import pandas as pd
import hashlib
from langchain.schema import Document
import os
import nltk
nltk.download("punkt")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["HF_API_KEY"] = st.secrets["HF_API_KEY"]

query_params = st.query_params
if "ping" in query_params and query_params["ping"][0].lower() == "true":
    st.write("Ping received — app is alive.")
    st.stop()

st.set_page_config(page_title="TutorMind - AI Tutor")
st.title("TutorMind - Personalized AI Tutor")


st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        margin-bottom: 2rem !important;
    }

    header[data-testid="stHeader"] {
        height: 0rem;
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔐 User Access")

if "user_id" not in st.session_state:
    # if st.session_state.get("was_logged_in"):
    #     st.info("⚠️ You've been logged out (e.g. due to refresh). Please log in again.")
    #     st.session_state["was_logged_in"] = False

    auth_mode = st.sidebar.radio("Choose Mode", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Register":
        if st.sidebar.button("Create Account"):
            success, msg = register_user(username, password)
            if success:
                st.sidebar.success("Success! Account created. Please log in.")
            else:
                st.sidebar.error(f"Failed! {msg}")
            st.stop()

    elif auth_mode == "Login":
        if st.sidebar.button("Login"):
            if validate_user(username, password):
                st.session_state.clear()
                st.session_state["user_id"] = username
                st.session_state["login_success"] = True
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
                st.stop()
else:
    st.sidebar.success(f"Logged in as: {st.session_state['user_id']}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        del st.session_state["user_id"]
        st.session_state["was_logged_in"] = True
        st.rerun()
    with st.sidebar.expander("🔄 Change Password"):
        old_pw = st.text_input("Current password", type="password")
        new_pw = st.text_input("New password", type="password")
        if st.button("Update Password"):
            from backend.firebase_auth import change_password
            success, msg = change_password(st.session_state["user_id"], old_pw, new_pw)
            if success:
                st.success(msg)
            else:
                st.error(f"Failed! {msg}")

if "user_id" not in st.session_state:
    st.warning("Please log in to access the app.")
    st.stop()

user_id = st.session_state["user_id"]


model_info = {
    "llama3-8b-8192": {
        "Provider": "Groq",
        "Size": "8B",
        "License": "Meta LLAMA 3 Community License",
        "Link": "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
    },
    "llama3-70b-8192": {
        "Provider": "Groq",
        "Size": "70B",
        "License": "Meta LLAMA 3 Community License",
        "Link": "https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct"
    },
    "meta-llama/llama-3-8b-instruct": {
        "Provider": "OpenRouter",
        "Size": "8B",
        "License": "Meta LLAMA 3 Community License",
        "Link": "https://openrouter.ai/models/meta-llama/llama-3-8b-instruct"
    },
    "mistralai/mistral-7b-instruct": {
        "Provider": "OpenRouter",
        "Size": "7B",
        "License": "Apache 2.0",
        "Link": "https://openrouter.ai/models/mistralai/mistral-7b-instruct"
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "Provider": "HuggingFace",
        "Size": "7B",
        "License": "Apache 2.0",
        "Link": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1"
    },
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo": {
        "Provider": "OpenRouter",
        "Size": "Mixtral (MoE)",
        "License": "Apache 2.0",
        "Link": "https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral"
    },
    "openchat/openchat-3.5-1210": {
        "Provider": "OpenRouter",
        "Size": "7B",
        "License": "MIT",
        "Link": "https://openrouter.ai/models/openchat/openchat-3.5-1210"
    }
}

provider = st.selectbox("Model Provider:", ["groq", "openrouter", "huggingface"])

if provider == "groq":
    model_choice = st.selectbox("Choose your model:", ["llama3-8b-8192", "llama3-70b-8192"])
elif provider == "openrouter":
    model_choice = st.selectbox("Choose your model:", [
  "meta-llama/llama-3-8b-instruct",
  "mistralai/mistral-7b-instruct"
])
elif provider == "huggingface":
    model_choice = st.selectbox("Choose your model:", ["mistralai/Mistral-7B-Instruct-v0.1"])

st.session_state["provider"] = provider

with st.expander("📄 Model Info"):
    info = model_info.get(model_choice)
    if info:
        st.markdown(f"""
        **Model:** `{model_choice}`  
        **Provider:** `{info['Provider']}`  
        **Size:** `{info['Size']}`  
        **License:** `{info['License']}`  
        **[View Model]({info['Link']})**
        """)
    else:
        st.info("No model info available.")

model_comparison = pd.DataFrame([
    {"Model": "llama3-8b-8192", "Provider": "Groq", "Size": "8B", "Use Case": "Fast + accurate", "Speed": "⚡"},
    {"Model": "llama3-70b-8192", "Provider": "Groq", "Size": "70B", "Use Case": "Deep reasoning & analysis", "Speed": "⚡"},
    {"Model": "meta-llama/llama-3-8b-instruct", "Provider": "OpenRouter", "Size": "8B", "Use Case": "Balanced chat + QA", "Speed": "⚡"},
    {"Model": "mistralai/mistral-7b-instruct", "Provider": "OpenRouter", "Size": "7B", "Use Case": "Lightweight, general QA", "Speed": "⚡"},
    {"Model": "mistralai/Mistral-7B-Instruct-v0.1", "Provider": "HuggingFace", "Size": "7B", "Use Case": "Open-source QA + summarization", "Speed": "⚠️ Slower"}
])


with st.expander("📊 Model Comparison Table"):
    st.dataframe(model_comparison)

qa_mode = st.sidebar.radio("Q&A Mode", ["Conceptual (Generative)", "Factual (Extractive)"])
st.session_state["qa_mode"] = qa_mode


st.sidebar.markdown("### 🧹 Data Management")

clear_qa = st.sidebar.button("🗑️  Clear Q&A History", use_container_width=True)
clear_upload = st.sidebar.button("📁  Reset Upload History", use_container_width=True)

if clear_qa:
    clear_history(user_id)
    st.session_state.pop("qa_history", None)
    st.toast("❌ Q&A history cleared")

if clear_upload:
    for key in ["uploaded_hashes", "faiss_built", "vectorstore", "qa_chain", "last_model", "last_mode"]:
        st.session_state.pop(key, None)
    st.toast("🧼 Upload history cleared. Please re-upload files.")






if st.sidebar.button("♻️ Clear Streamlit Cache", use_container_width=True):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.toast("Cache cleared! Please re-upload files if needed.")

st.sidebar.markdown("---")
st.sidebar.subheader("📣 We value your feedback")

overall_rating = st.sidebar.slider("How would you rate TutorMind?", 1, 5, 4, key="overall_rating")
feedback_text = st.sidebar.text_area("Any comments or suggestions?", key="feedback_text")

if st.sidebar.button("Submit Feedback"):
    from backend.firebase_db import submit_app_feedback
    submit_app_feedback(st.session_state["user_id"], overall_rating, feedback_text)
    st.toast("Thank you for your feedback!")


def file_hash(file): 
    return hashlib.md5(file.getvalue()).hexdigest()

uploaded_files = st.file_uploader(
    "Upload documents (PDF/DOCX/TXT/IMG)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)


if "processed_file_hashes" not in st.session_state:
    st.session_state["processed_file_hashes"] = set()

if uploaded_files:
    uploaded_hashes = [file_hash(f) for f in uploaded_files]
    prev_hashes = st.session_state.get("uploaded_hashes", [])

    if uploaded_hashes != prev_hashes:
        for key in ["faiss_built", "vectorstore", "qa_chain", "last_model", "last_mode", "uploaded_hashes"]:
            st.session_state.pop(key, None)

        with st.spinner("Creating chunks and indexing documents..."):
            all_chunks = []

            for file in uploaded_files:
                hash_val = file_hash(file)
                if hash_val in st.session_state["processed_file_hashes"]:
                    continue

                text = load_any_file(file)
                if not text.strip():
                    st.warning(f"No extractable text found in file: {file.name}")
                    continue

                with st.expander(f"📄 Preview Extracted Content: `{file.name}`"):
                    preview_lines = text.strip().splitlines()[:30]
                    preview = "\n".join(preview_lines)
                    st.markdown(f"```text\n{preview}\n```")

                chunks = adaptive_chunk_text(text)
                for chunk in chunks:
                    all_chunks.append(Document(page_content=chunk, metadata={"source": file.name}))

                st.session_state["processed_file_hashes"].add(hash_val)

            if not all_chunks:
                st.error("No text chunks found. Please upload a valid document.")
                st.stop()

            create_faiss_index(all_chunks)
            st.session_state["all_chunks"] = all_chunks
            st.session_state["faiss_built"] = True
            st.session_state["uploaded_hashes"] = uploaded_hashes  
            st.success("All documents embedded and indexed!")


if "vectorstore" not in st.session_state:
    if st.session_state.get("faiss_built"):
        st.session_state["vectorstore"] = load_faiss_index()
    else:
        st.warning("Please upload valid documents to build the vector store.")
        st.stop()

if "last_mode" not in st.session_state or st.session_state["last_mode"] != qa_mode \
   or "last_model" not in st.session_state or st.session_state["last_model"] != model_choice:

    if "vectorstore" in st.session_state:
        try:
            st.session_state["qa_chain"] = build_qa_chain(
                vectorstore=st.session_state["vectorstore"],
                model_choice=model_choice,
                mode=qa_mode,
                provider=provider
            )
        except Exception as e:
            st.error(f"Failed to load model `{model_choice}`. Error: {str(e)}")
            st.stop()
        st.session_state["last_mode"] = qa_mode
        st.session_state["last_model"] = model_choice



st.markdown(f"""
<div style='display: flex; gap: 0.5rem; margin-bottom: 1rem;'>
    <span style='background-color: #262730; color: white; padding: 0.3rem 0.6rem; border-radius: 0.5rem; font-size: 0.85rem;'>🧠 Model: <strong>{model_choice}</strong></span>
    <span style='background-color: #1c1c1e; color: white; padding: 0.3rem 0.6rem; border-radius: 0.5rem; font-size: 0.85rem;'>⚙️ Mode: <strong>{qa_mode}</strong></span>
    <span style='background-color: #373737; color: white; padding: 0.3rem 0.6rem; border-radius: 0.5rem; font-size: 0.85rem;'>🔌 Provider: <strong>{provider}</strong></span>
</div>
""", unsafe_allow_html=True)



tab1, tab2, tab3 = st.tabs(["💬 Ask", "📜 History", "📊 Insights"])

with tab1:
    st.subheader("💬 Ask a question from your uploaded material")

    all_chunks = st.session_state.get("all_chunks", [])
    if all_chunks:
        sources = list({doc.metadata.get("source", "Unknown") for doc in all_chunks})
        selected_files = st.multiselect("📂 Ask from specific file(s)", sources, default=sources)
        st.session_state["selected_files"] = selected_files
    else:
        st.info("No documents available yet. Please upload.")

    query = st.text_input("Enter your question:", key="user_query")

    if query:
        result = st.session_state["qa_chain"](query)
        with st.spinner("Thinking..."):
            st.markdown("### Answer:")
            st.markdown(result["result"])
            with st.expander("📄 Source Context"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"📁 **File:** `{doc.metadata.get('source', 'Unknown')}`")
                    st.markdown(f"**Chunk {i+1}:**\n```\n{doc.page_content}\n```")
            save_qa(user_id, query, result["result"])

with tab2:
    st.subheader("📜 Your Q&A History")
    search_query = st.text_input("Search previous question", key="search_q")
    user_history = get_history(user_id)
    if user_history:
        filtered = [qa for qa in user_history if search_query.lower() in qa["question"].lower()] if search_query else user_history
        for i, qa in enumerate(filtered[::-1]):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.markdown(f"**Ans:** {qa['answer']}")

with tab3:
    st.subheader("📊 Session Insights")
    if user_history:
        df = pd.DataFrame(user_history)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions Asked", len(df))
            st.metric("Avg Answer Length", round(df['answer'].str.len().mean(), 1))
        with col2:
            longest = df.loc[df['answer'].str.len().idxmax()]
            st.metric("Longest Answer", f"{len(longest['answer'])} chars")
        with st.expander("📋 View Data Table"):
            st.dataframe(df)


