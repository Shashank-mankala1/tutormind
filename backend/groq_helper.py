# backend/groq_helper.py
import requests
import os
import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_groq_key():
    return GROQ_API_KEY or os.environ.get("GROQ_API_KEY") or ("st.secrets" in globals() and st.secrets.get("GROQ_API_KEY"))

def query_groq(prompt, model="llama3-8b-8192"):
    api_key = get_groq_key()
    if not api_key:
        raise Exception("GROQ_API_KEY not set in environment or Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, json=body)

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code}\n{response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()
