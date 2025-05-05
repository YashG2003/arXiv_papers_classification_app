import re
import io
import PyPDF2

def extract_from_pdf(pdf_file):
    """Extract using PyPDF2"""
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Get text from first page
    first_page_text = pdf_reader.pages[0].extract_text()
    
    # Extract title - assume it's in the first few lines
    lines = first_page_text.split('\n')
    # Combine first 2-3 lines to get title
    title_lines = [line.strip() for line in lines[:4] if line.strip()]
    title = " ".join(title_lines)
    
    # Extract abstract - look for text between "Abstract" and the next section
    full_text = ""
    for i in range(min(2, len(pdf_reader.pages))):  # Look in first 2 pages
        full_text += pdf_reader.pages[i].extract_text()
    
    abstract_match = re.search(r'Abstract(.*?)(?:Introduction|Keywords|I\.|1\.)', 
                              full_text, re.DOTALL | re.IGNORECASE)
    
    if abstract_match:
        abstract = abstract_match.group(1).strip()
    else:
        # If no clear markers, take text after title until a reasonable length
        abstract = first_page_text.replace(title, "", 1).strip()
        # Limit to first 2000 characters as a reasonable abstract length
        abstract = abstract[:500]
    
    return title, abstract
