import os
import streamlit as st
import json

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["HF_API_KEY"] = st.secrets["HF_API_KEY"]

def query_model(prompt, provider="groq", model="llama3-8b-8192"):
    if provider == "groq":
        from backend.groq_helper import query_groq
        return query_groq(prompt, model=model)

    elif provider == "openrouter":
        import requests
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://your-app.com/",
            "X-Title": "TutorMind"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        # return res.json()["choices"][0]["message"]["content"]
        try:
            res_json = res.json()
            first_choice = res_json["choices"][0]

            if "choices" not in res_json:
                raise ValueError(f"OpenRouter response missing 'choices': {res.text}")
            if "message" in first_choice and "content" in first_choice["message"]:
                return first_choice["message"]["content"]
            elif "text" in first_choice:
                return first_choice["text"]
            else:
                raise ValueError(f"Unexpected OpenRouter response format: {res.text}")
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from OpenRouter: {res.text}")

    elif provider == "huggingface":
        import requests
        HF_API_KEY = os.getenv("HF_API_KEY")
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        data = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512}
        }
        url = f"https://api-inference.huggingface.co/models/{model}"
        res = requests.post(url, headers=headers, json=data)

        if res.status_code != 200:
            raise Exception(f"HuggingFace API Error {res.status_code}: {res.text}")

        return res.json()[0]["generated_text"]

    elif provider == "local":
        from backend.local_llm import LocalLLM
        return LocalLLM(model_id=model)(prompt)

    else:
        raise ValueError("Unknown provider")
