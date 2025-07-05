# 🎓 TutorMind – AI-Powered Personalized Tutor

TutorMind is a document-grounded GenAI assistant that transforms your files into an intelligent tutor. Powered by Retrieval-Augmented Generation (RAG), LangChain, and multiple LLMs, TutorMind helps users ask **precise, contextual questions** from **PDFs, DOCX, TXT, or even image-based content**.

---

## 🚀 What's New in Version 2.0

* 🔁 **Multi-Model Backend**: Supports Groq, OpenRouter, HuggingFace, and Local inference with dynamic routing
* 📁 **Multi-File Uploads**: Supports PDF, DOCX, TXT, and images with OCR fallback
* 📂 **File-Specific QA**: Select files to ask questions from and see file-specific source context
* 🧠 **Dual QA Modes**: Switch between Conceptual (LLM) and Factual (BERT extractive) answering
* 🔍 **Adaptive Chunking**: Uses clustering on sentence embeddings for intelligent segmentation
* 📊 **User Insights**: Tracks questions asked, average answer length, and session stats
* 🗳️ **Feedback System**: Rate your experience and submit optional comments
* ✅ **Improved UI/UX**: Tabbed layout, clean session handling, and model info comparison

---

## 📈 Tech Stack

| Layer      | Stack                                    |
| ---------- | ---------------------------------------- |
| Frontend   | Streamlit                                |
| Backend    | Python + LangChain                       |
| LLMs       | Groq, OpenRouter, HuggingFace, Local     |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB  | FAISS                                    |
| Auth + DB  | Firebase Firestore                       |

---

## 🤖 How It Works

1. **User Login/Register** (Firestore-auth)
2. **Upload files** (PDF/DOCX/TXT/Images)
3. **Text extracted + chunked** using ML-based adaptive chunking
4. **Embeddings stored** in FAISS
5. **Question asked → RAG pipeline** triggered
6. **Answer generated** using chosen LLM (Factual or Generative)
7. **Answer & source chunks returned**
8. **Session saved per user in Firestore**

---

## 🧠 Model Architecture

| Mode       | Model Examples                            | Use Case                           |
| ---------- | ----------------------------------------- | ---------------------------------- |
| Extractive | BERT (deepset/squad2)                     | Direct factual Qs                  |
| Generative | LLaMA3, Mistral, Flan-T5, OpenChat, etc.  | Why/How/Conceptual Qs              |
| Backend    | Dynamic `query_model()` abstraction layer | Unified across Groq/OpenRouter/etc |

---

## 🔐 Authentication

* 🔑 Firebase-backed login and registration
* 🔒 Passwords securely hashed (SHA-256)
* 🔁 Password change support
* 🧠 Session-based state with Streamlit

---

## ✨ Key Features

* 📁 Upload any document (PDF, DOCX, TXT, IMG)
* 🧠 Ask deep or factual questions from your content
* 📂 Ask from specific files only
* 📄 View exact source context with file label
* 🔁 Clear Q\&A history or uploaded files
* 🧠 Model + QA Mode switching (Groq / OpenRouter / HuggingFace)
* 📊 See insights on session activity (e.g., longest answer)
* 🗣️ Submit feedback directly
* ✅ Fully deployable on Streamlit Cloud

---

## 🧪 Future Enhancements

* 🗨️ Real-time Chat Mode with memory
* 📈 Admin dashboard & analytics
* 🎯 Adaptive reranking based on feedback
* 📤 Export sessions to PDF or shareable link
* 🗣️ Whisper voice input

---

## 📘 License

MIT License – Free for personal, academic, or educational use.

---

## ✨ Made with ❤️ by Shashank using GenAI, LangChain, and Streamlit
