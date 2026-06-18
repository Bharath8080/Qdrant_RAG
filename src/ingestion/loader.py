# This file reads a PDF file and returns all the text from it

import fitz  # PyMuPDF


def load_pdf(file) -> str:
    """
    Takes a PDF file (from Streamlit uploader) and returns all its text using PyMuPDF.
    
    Args:
        file: the uploaded file object from Streamlit
    
    Returns:
        A single string with all the text from every page
    """
    # Ensure file pointer is at the beginning
    file.seek(0)
    pdf_bytes = file.read()
    
    # Open the PDF directly from the byte stream (in-memory, no temp files needed)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    all_text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            all_text += page_text + "\n"
            
    return all_text


