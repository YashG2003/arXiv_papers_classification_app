# app/frontend/app.py
import streamlit as st
import requests
import pandas as pd
import time

# Mapping from code to human-readable name
CATEGORY_MAP = {
    'astro-ph': 'Astrophysics',
    'cond-mat': 'Condensed Matter Physics',
    'cs': 'Computer Science',
    'eess': 'Electrical Engineering and Systems Science',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math': 'Mathematics',
    'physics': 'Physics (General)',
    'quant-ph': 'Quantum Physics',
    'stat': 'Statistics'
}
# Reverse mapping from human-readable name to code
NAME_TO_CODE = {v: k for k, v in CATEGORY_MAP.items()}

# API configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="arXiv Paper Classifier",
    page_icon="📚",
    layout="wide"
)

# Custom CSS with improved visibility
st.markdown("""
<style>
    .paper-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4285F4;
        color: #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .correct-prediction {
        border-left: 4px solid #34A853;
    }
    .incorrect-prediction {
        border-left: 4px solid #EA4335;
    }
    .paper-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    .paper-abstract {
        font-size: 0.9em;
        color: #555;
    }
    .category-label {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.8em;
        margin-left: 5px;
    }
    .true-label {
        background-color: #d4edda;
        color: #155724;
    }
    .pred-label {
        background-color: #cce5ff;
        color: #004085;
    }
    .prediction-header {
        padding: 15px;
        background-color: #e3f2fd;
        border-radius: 5px;
        margin-bottom: 15px;
        color: #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.title("📚 arXiv Paper Classifier")
st.write("Classify research papers into 10 categories based on their title and abstract.")

# Function to get samples
def get_samples():
    try:
        response = requests.get(f"{API_URL}/samples")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Create tabs for input and samples
tab1, tab2 = st.tabs(["📄 Upload Paper", "🔍 Sample Predictions"])

# First tab - PDF upload
with tab1:
    st.subheader("Upload a Research Paper PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Extracting and classifying..."):
            # Send the PDF file to the backend for extraction and classification
            files = {"file": uploaded_file}
            response = requests.post(f"{API_URL}/classify_pdf", files=files)
            
            if "error" in response.json():
                st.error(f"Error: {response.json()['error']}")
            else:
                result = response.json()
                
                # Initialize session state for correction mode if it doesn't exist
                if 'correction_mode' not in st.session_state:
                    st.session_state.correction_mode = False
                
                # Display the result
                st.success("Classification complete!")
                
                # Create a nice result box with improved visibility
                st.markdown(f"""
                <div class="paper-card">
                    <h3 style="color: #333;">Predicted Category: {result['category']}</h3>
                    <p style="color: #555;">Confidence: {result['confidence']:.4f}</p>
                    <p style="color: #555;">Processing Time: {result['processing_time']:.3f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Only show feedback options if not in correction mode
                if not st.session_state.correction_mode:
                    st.write("")
                    st.write("Is this prediction correct?")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Yes, it's correct"):
                            st.success("Thank you for your feedback!")
                    
                    with col2:
                        if st.button("❌ No, it's incorrect"):
                            # Store information for correction
                            st.session_state.correction_mode = True
                            st.session_state.title = result['title'] 
                            st.session_state.abstract = result['abstract']
                            st.session_state.predicted_category = result['category']
                            # Force rerun to show correction form
                            st.rerun()
                
                # Show correction form if in correction mode
                if st.session_state.correction_mode:
                    st.write("")
                    st.subheader("Please select the correct category:")
                    
                    categories = list(CATEGORY_MAP.values())
                    
                    correct_category = st.selectbox("Correct Category", categories)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Submit Correction"):
                            # Submit the feedback to the backend
                            feedback_data = {
                                "title": st.session_state.title,
                                "abstract": st.session_state.abstract,
                                "predicted_category": st.session_state.predicted_category,
                                "correct_category": NAME_TO_CODE[correct_category]
                            }
                            
                            try:
                                response = requests.post(f"{API_URL}/submit_feedback", json=feedback_data)
                                response_data = response.json()
                                
                                if response.status_code == 200:
                                    st.success(f"Thank you for your feedback! ({response_data['feedback_count']}/{response_data['threshold']} entries collected)")
                                    # Clear the correction state
                                    st.session_state.correction_mode = False
                                    st.rerun()
                                else:
                                    st.error(f"Error submitting feedback: {response_data.get('detail', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    with col2:
                        if st.button("Cancel"):
                            st.session_state.correction_mode = False
                            st.rerun()


# Second tab - Sample predictions
with tab2:
    st.subheader("Sample Predictions from Test Set")
    
    if st.button("Load Sample Predictions"):
        with st.spinner("Loading samples..."):
            samples_data = get_samples()
            
            if "error" in samples_data:
                st.error(f"Error loading samples: {samples_data['error']}")
            else:
                # Show overall accuracy with improved visibility
                st.markdown(f"""
                <div class='prediction-header'>
                    <h3 style="color: #333; margin: 0;">Overall Accuracy: {samples_data['accuracy']:.2%}</h3>
                    <p style="color: #555; margin: 5px 0 0 0;">Total Samples: {samples_data['total_count']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display samples by category - reformatted to show just title
                for category, samples in samples_data['samples_by_category'].items():
                    with st.expander(f"{category} ({len(samples)} papers)", expanded=True):
                        for sample in samples:
                            # Determine if prediction is correct
                            css_class = "correct-prediction" if sample["correct"] else "incorrect-prediction"
                            
                            # Create paper card with improved layout
                            st.markdown(f"""
                            <div class="paper-card {css_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: bold; color: #333;">Predicted: {sample['predicted_category']}</span>
                                    <span class="category-label true-label">True: {sample['true_category']}</span>
                                </div>
                                <div class="paper-title" style="margin-top: 8px;">{sample['title']}</div>
                            </div>
                            """, unsafe_allow_html=True)
