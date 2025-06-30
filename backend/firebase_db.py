import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

# Load credentials
key_path = os.path.join("backend", "firebase_key.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_qa(user_id, question, answer):
    user_ref = db.collection("users").document(user_id).collection("qa_history")
    user_ref.add({"question": question, "answer": answer, "timestamp": datetime.utcnow()
    })


def get_history(user_id):
    user_ref = db.collection("users").document(user_id).collection("qa_history")
    docs = user_ref.stream()
    return [{"question": doc.to_dict()["question"], "answer": doc.to_dict()["answer"]} for doc in docs]

def clear_history(user_id):
    user_ref = db.collection("users").document(user_id).collection("qa_history")
    docs = user_ref.stream()
    for doc in docs:
        user_ref.document(doc.id).delete()

def submit_app_feedback(user_id, rating, comment):
    """Stores general app feedback from user"""
    feedback_ref = db.collection("feedback")
    feedback_ref.add({
        "user_id": user_id,
        "rating": rating,
        "comment": comment,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
