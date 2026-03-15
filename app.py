import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="AI Resume Screening System", page_icon="🤖", layout="wide")

def clean_text(text):
    """Clean text by removing extra whitespace and converting to lowercase."""
    if text:
        text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def calculate_similarity(resume_text, job_desc_text):
    """Calculate similarity score using TF-IDF and cosine similarity."""
    if not resume_text or not job_desc_text:
        return 0.0
    
    documents = [resume_text, job_desc_text]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100

def get_match_message(score):
    """Get match message based on score."""
    if score > 70:
        return "✅ **Excellent Match!** Strong candidate for the role."
    elif score > 50:
        return "✅ **Good Match!** Potential candidate."
    elif score > 30:
        return "⚠️ **Moderate Match.** Needs further review."
    else:
        return "❌ **Low Match.** May not be suitable for this role."

# Header
st.title("🤖 AI Resume Screening System")
st.markdown("---")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Paste your **resume text** in the left box
    2. Paste the **job description** in the right box  
    3. Click **Screen Resume** to get instant match score!
    """)
    st.markdown("---")
    st.header("⚙️ Features")
    st.markdown("- TF-IDF Vectorization")
    st.markdown("- Cosine Similarity Matching") 
    st.markdown("- Real-time scoring")

# Main content - two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Resume")
    resume_text = st.text_area(
        "Paste resume content here:",
        height=300,
        placeholder="Your skills, experience, education..."
    )

with col2:
    st.subheader("📋 Job Description") 
    job_desc_text = st.text_area(
        "Paste job description here:",
        height=300,
        placeholder="Required skills, responsibilities, requirements..."
    )

# Calculate button
if st.button("🚀 Screen Resume", type="primary"):
    with st.spinner("Analyzing resume match..."):
        cleaned_resume = clean_text(resume_text)
        cleaned_job = clean_text(job_desc_text)
        
        match_score = calculate_similarity(cleaned_resume, cleaned_job)
        
        # Results
        col_result1, col_result2 = st.columns([1, 3])
        
        with col_result1:
            st.metric("Match Score", f"{match_score:.1f}%")
        
        with col_result2:
            st.markdown(get_match_message(match_score))
        
        st.markdown("---")
        
        # Progress bar
        st.progress(match_score / 100)
        
        # Color-coded gauge
        gauge_color = "green" if match_score > 70 else "orange" if match_score > 50 else "red"
        st.markdown(
            f"""
            <div style="background: linear-gradient(to right, 
                red 0%, red {30}%, orange {30}%, orange {50}%, 
                yellow {50}%, yellow {70}%, green {70}%, green 100%); 
                width: 100%; height: 20px; border-radius: 10px;">
                <div style="background-color: {gauge_color}; 
                    width: {match_score}%; height: 20px; border-radius: 10px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Examples button
if st.button("📝 Load Sample Data"):
    st.session_state.sample_loaded = True

if st.session_state.get('sample_loaded', False):
    with col1:
        st.text_area("Resume sample loaded!", value="""John Doe
Software Engineer with 5+ years Python/Django experience
Expertise in ML with scikit-learn, AWS deployment""", key="sample_resume")
    
    with col2:
        st.text_area("Job sample loaded!", value="""Senior Python Developer
Django, scikit-learn, AWS, ML model development required""", key="sample_job")

st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit + scikit-learn**")

