
import io
import re
from typing import List, Dict
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def _normalize_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def process_uploaded_file(uploaded_file) -> List[Dict]:
    """
    Process an uploaded file and return a list of text chunks.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        
    Returns:
        List[Dict]: List of chunks with 'text' and 'metadata'.
    """
    filename = uploaded_file.name
    file_type = uploaded_file.type
    
    text = ""
    
    try:
        if file_type == "application/pdf":
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        elif file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            
        else:
            return [] # Unsupported type
            
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return []
    
    text = _normalize_extracted_text(text)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    # Format as document dicts
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{filename}_{i}",
            "text": chunk,
            "metadata": {
                "source": filename,
                "chunk_id": i
            }
        })
        
    return documents
