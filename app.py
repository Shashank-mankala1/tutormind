import streamlit as st
from backend.loader import load_any_file
from backend.qa_chain import load_faiss_index, build_qa_chain
from backend.embedder import chunk_text, create_faiss_index
from backend.firebase_db import save_qa, get_history
import pandas as pd
from backend.firebase_db import clear_history
from backend.firebase_auth import register_user, validate_user
from backend.firebase_db import submit_app_feedback

st.set_page_config(page_title="TutorMind - AI Tutor")
st.title("TutorMind - Personalized AI Tutor")


st.sidebar.title("🔐 User Access")

if "user_id" not in st.session_state:
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
                st.session_state["user_id"] = username
                st.session_state["login_success"] = True
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
                st.stop()
else:
    st.sidebar.success(f"Logged in as: {st.session_state['user_id']}")
    if st.sidebar.button("🚪 Logout"):
        del st.session_state["user_id"]
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


model_choice = st.selectbox("Choose your model:", [
    "google/flan-t5-base",
    "google/flan-t5-small",
    "google/flan-t5-large",
    "declare-lab/flan-alpaca-base",
    "MBZUAI/LaMini-Flan-T5-783M"
])

auto_mode = st.sidebar.checkbox("Auto-detect Q&A mode", value=True)

if auto_mode:
    def detect_mode(question):
        if any(word in question.lower() for word in ["who", "when", "where", "email", "contact", "name", "instructor", "worth"]):
            return "Factual (Extractive)"
        return "Conceptual (Generative)"
    qa_type = detect_mode(st.session_state.get("user_query", ""))
else:
    qa_type = st.sidebar.radio("Choose Q&A mode:", ["Conceptual (Generative)", "Factual (Extractive)"])
    st.sidebar.info("Conceptual = summary/explanation\n Factual = direct name/email/date answers")

st.session_state["model_choice"] = model_choice

if st.sidebar.button("🗑️ Clear Q&A History"):
    clear_history(user_id)
    if "qa_history" in st.session_state:
        del st.session_state["qa_history"]
    st.success("Your Q&A history has been cleared!")

st.sidebar.markdown("---")
st.sidebar.subheader("📣 We value your feedback")

overall_rating = st.sidebar.slider("How would you rate TutorMind?", 1, 5, 4, key="overall_rating")
feedback_text = st.sidebar.text_area("Any comments or suggestions?", key="feedback_text")

if st.sidebar.button("Submit Feedback"):
    from backend.firebase_db import submit_app_feedback
    submit_app_feedback(st.session_state["user_id"], overall_rating, feedback_text)
    st.toast("Thank you for your feedback!")


uploaded_files = st.file_uploader(
    "Upload your curriculum files (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files and "faiss_built" not in st.session_state:
    all_text = ""
    with st.spinner("Processing uploaded files..."):
        for file in uploaded_files:
            text = load_any_file(file)
            all_text += text + "\n"

        chunks = chunk_text(all_text)
        create_faiss_index(chunks)
        st.session_state["faiss_built"] = True
        st.success("All documents embedded and indexed!")

# Load vectorstore
if "vectorstore" not in st.session_state and "faiss_built" in st.session_state:
    st.session_state["vectorstore"] = load_faiss_index()

# Build QA chain with selected model
if "last_mode" not in st.session_state or st.session_state["last_mode"] != qa_type \
   or "last_model" not in st.session_state or st.session_state["last_model"] != model_choice:
    
    if "vectorstore" in st.session_state:
        st.session_state["qa_chain"] = build_qa_chain(
            st.session_state["vectorstore"],
            model_choice=model_choice,
            mode=qa_type
        )
        st.session_state["last_mode"] = qa_type
        st.session_state["last_model"] = model_choice



# Q&A Interaction
if "qa_chain" in st.session_state:
    if "query_submitted" not in st.session_state:
        st.session_state.query_submitted = False

    if "suggested_query" in st.session_state:
            st.session_state["user_query"] = st.session_state["suggested_query"]
            del st.session_state["suggested_query"] 

    st.subheader("💬 Ask a question from your uploaded material")


    query = st.text_input("Enter your question", key="user_query")
    
    if "qa_history" in st.session_state and st.session_state["qa_history"]:
        st.markdown("**Recent Questions:**")
        for i, qa in enumerate(st.session_state["qa_history"][-3:][::-1]):
            if st.button(f"🔁 {qa['question']}", key=f"recent_q_{i}"):
                st.session_state["suggested_query"] = qa["question"]
                st.experimental_rerun()

    if query and not st.session_state.query_submitted:
        st.session_state.query_submitted = True
        with st.spinner("Thinking..."):
            result = st.session_state["qa_chain"](query)

            if "bad person" in result["result"].lower():
                st.warning("The model generated inappropriate or irrelevant text.")
                st.info("💡 Tip: Switch to a more accurate model like `LaMini-Flan-T5-783M` for better results.")
            else:
                st.markdown(f"""
                <div style='padding: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem; border-radius: 12px;
                            border: 1px solid #666; background-color: rgba(255, 255, 255, 0.05);
                            color: inherit;'>

                <h4 style='margin-bottom: 0.2rem;'> Answer:</h4>
                {result["result"]}

                </div>
                """, unsafe_allow_html=True)

            # Source chunks
            with st.expander("📄 Show Source Context"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")

            save_qa(user_id, query, result["result"])

        st.session_state.query_submitted = False





# Display Firebase Q&A history
st.markdown("---")
st.subheader("🔎 Search your past questions:")
search_query = st.text_input("Enter your question", key="search_q")

user_history = get_history(user_id)
if user_history:
    if search_query:
        filtered = [qa for qa in user_history if search_query.lower() in qa["question"].lower()]
    else:
        filtered = user_history

    st.markdown("### 📝 Your Q&A History")
    for i, qa in enumerate(filtered[::-1]):
        with st.expander(f"Q{i+1}: {qa['question']}"):
            st.markdown(f"**Ans:** {qa['answer']}")


user_data = get_history(user_id)
if user_data:
    df = pd.DataFrame(user_data)

    st.markdown("### 📊 Your Session Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Questions Asked", len(df))
        st.metric("Avg Answer Length", round(df['answer'].str.len().mean(), 1))

    with col2:
        longest = df.loc[df['answer'].str.len().idxmax()]
        st.metric("Longest Answer", f"{len(longest['answer'])} chars")

    with st.expander("📋 View Data Table"):
        st.dataframe(df)