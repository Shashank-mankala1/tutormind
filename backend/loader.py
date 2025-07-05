import PyPDF2
import docx
from io import BytesIO
import chardet
import requests
import streamlit as st
import hashlib

OCR_SPACE_API_KEY = st.secrets["OCR_SPACE_API_KEY"]

def file_hash(file):
    file.seek(0)
    return hashlib.md5(file.read()).hexdigest()

def call_ocr_space_pdf(file_bytes):
    url = 'https://api.ocr.space/parse/image'
    files = {
        'filename': ('document.pdf', file_bytes, 'application/pdf')
    }
    payload = {
        'isOverlayRequired': False,
        'apikey': OCR_SPACE_API_KEY,
        'language': 'eng',
        'filetype': 'pdf'
    }
    response = requests.post(url, files=files, data=payload)
    result = response.json()

    if result.get('IsErroredOnProcessing') or not result.get('ParsedResults'):
        return ""

    return result['ParsedResults'][0].get('ParsedText', '').strip()

def split_and_ocr_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        output_text = ""
        for i, page in enumerate(reader.pages):
            writer = PyPDF2.PdfWriter()
            writer.add_page(page)
            with BytesIO() as buffer:
                writer.write(buffer)
                single_page_bytes = buffer.getvalue()
                page_text = call_ocr_space_pdf(single_page_bytes)
                if page_text:
                    output_text += page_text + "\n"
        if output_text.strip():
            st.toast("OCR fallback used — scanned document processed.")
            return output_text.strip()
        else:
            return "[OCR fallback failed: no text extracted from any page]"
    except Exception as e:
        return f"[OCR fallback exception: {e}]"

def load_pdf(file):
    try:
        hash_val = file_hash(file)
        if hash_val in st.session_state.processed_file_hashes:
            return ""
        file.seek(0)
        file_bytes = file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        if not text.strip():
            result = split_and_ocr_pdf(file_bytes)
        else:
            result = text

        st.session_state.processed_file_hashes.add(hash_val)
        return result
    except Exception as e:
        return f"[Error reading PDF file: {e}]"

def load_docx(file):
    try:
        hash_val = file_hash(file)
        if hash_val in st.session_state.processed_file_hashes:
            return ""
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        st.session_state.processed_file_hashes.add(hash_val)
        return "\n".join(full_text)
    except Exception as e:
        return f"[Error reading DOCX file: {e}]"

def load_txt(file):
    try:
        hash_val = file_hash(file)
        if hash_val in st.session_state.processed_file_hashes:
            return ""
        file.seek(0)
        raw_bytes = file.read()
        detected = chardet.detect(raw_bytes)
        encoding = detected['encoding'] or 'utf-8'
        result = raw_bytes.decode(encoding, errors="ignore")
        st.session_state.processed_file_hashes.add(hash_val)
        return result
    except Exception as e:
        return f"[Error reading TXT file: {e}]"

def load_any_file(file):
    filename = file.name.lower()
    if filename.endswith(".pdf"):
        return load_pdf(file)
    elif filename.endswith(".docx"):
        return load_docx(file)
    elif filename.endswith(".txt"):
        return load_txt(file)
    elif file.type.startswith("image/"):
        import easyocr
        from PIL import Image
        import numpy as np

        reader = easyocr.Reader(['en'])
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)
        result = reader.readtext(image_np, detail=0)
        return "\n".join(result)

    else:
        return "[Unsupported file type]"
