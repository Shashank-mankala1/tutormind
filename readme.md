# 🎓 TutorMind – AI-Powered Personalized Tutor

TutorMind is an intelligent, interactive Q\&A system that transforms your study material into a private tutor using the power of Generative AI and Retrieval-Augmented Generation (RAG). It allows users to upload educational documents, ask questions, and receive high-quality, contextual answers with source references — all through a secure, personalized experience.


---

## 📈 Tech Stack

| Layer      | Stack                                    |
| ---------- | ---------------------------------------- |
| Frontend   | Streamlit                                |
| Backend    | Python + LangChain                       |
| LLMs       | Flan-T5, LaMini-Flan-T5, BERT            |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB  | FAISS                                    |
| Auth + DB  | Firebase Firestore                       |

---

## 🤖 How It Works

1. **User Login/Register** (Firestore-auth)
2. **Upload PDFs/DOCX/TXT**
3. **Documents are chunked + embedded** via MiniLM
4. **Embeddings stored** in FAISS vectorstore
5. **Question asked → RAG pipeline** triggered
6. **Answer generated** using BERT or Flan-T5 (auto-selected)
7. **Answer & source returned** + saved in Firestore
8. **History displayed** + can be cleared/exported

---

## 🎓 AI Model Logic

| Type       | Model                    | Use Case                  |
| ---------- | ------------------------ | ------------------------- |
| Extractive | BERT (deepset/squad2)    | Who/When/Where factual Qs |
| Generative | Flan-T5 / LaMini-Flan-T5 | Why/How/Open-ended Qs     |

> Auto-detection chooses model based on question type.

---

## 🔐 Authentication & User Management

* User registration with username + password
* Passwords hashed via **SHA-256** for security
* Users can **login**, **logout**, and **change password**
* Q\&A **history stored per user** in Firebase
* Session state managed by Streamlit

---


## 🔧 For Developers

* Modular structure with `app.py`, `qa_chain.py`, `firebase_auth.py`, `firebase_db.py`
* Easily extend with Whisper for voice input or new LLM endpoints (e.g., GPT, Claude)
* Add feedback, upvote/downvote answers, or export full session history

---

## 🔔 Use Cases

* 📚 Students querying syllabus/notes
* 💡 Corporate training assistants
* 📂 Document Q\&A for HR, SOPs, etc.

---

## ✨ Key Features

* 📂 **Upload Any Document**: PDF, DOCX, or TXT files supported
* 🧠 **Ask Questions**: Extractive or generative answers from your material
* 🔎 **Semantic Search**: Find previous Q\&A using keyword or context
* 🔐 **Secure User Accounts**: Firebase-backed registration, login, and password management
* ⏲️ **Session Handling**: Keeps Q\&A history per user, exportable to PDF
* ⚖️ **Auto Model Selection**: BERT for factual, Flan/LaMini for conceptual questions
* ♻️ **Q\&A History Control**: Clear history and manage session data
* ✅ **Deployable on Streamlit Cloud**
---


## 📊 Future Enhancements

* 🎤 Whisper voice-based Q\&A input
* 🧪 GPT / Claude integration with GGUF models
* 📄 Multi-Language Support

---

## 📘 License

MIT License – Free for personal, academic, or educational use.

---

## ✨ Made with ❤️ using GenAI + LangChain + Streamlit by Shashank
