import PyPDF2
import docx
from io import BytesIO

def load_pdf(file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def load_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def load_txt(file):
    return file.read().decode("utf-8")

def load_any_file(file):
    filename = file.name.lower()
    if filename.endswith(".pdf"):
        return load_pdf(file)
    elif filename.endswith(".docx"):
        return load_docx(file)
    elif filename.endswith(".txt"):
        return load_txt(file)
    else:
        return ""
