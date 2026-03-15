import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_text_file(file_path):
    """Load text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Clean the text: remove extra whitespace and convert to lowercase
            content = re.sub(r'\s+', ' ', content).strip().lower()
            return content
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None

def calculate_resume_match(resume_text, job_desc_text):
    """
    Calculate the similarity score between resume and job description using TF-IDF and cosine similarity.
    
    Args:
        resume_text (str): Cleaned resume content
        job_desc_text (str): Cleaned job description content
    
    Returns:
        float: Similarity score as percentage (0-100)
    """
    # Combine texts for vectorization
    documents = [resume_text, job_desc_text]
    
    # Create TF-IDF Vectorizer (ignores common English words and uses term frequency-inverse document frequency)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Convert texts to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity between resume (index 0) and job description (index 1)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to percentage
    match_percentage = similarity_score * 100
    
    return round(match_percentage, 2)

def main():
    """Main function to run the resume screening."""
    print("=== AI Resume Screening System ===")
    print()
    
    # Load resume and job description
    resume_content = load_text_file('resume.txt')
    job_desc_content = load_text_file('job_description.txt')
    
    if resume_content is None or job_desc_content is None:
        print("Please ensure resume.txt and job_description.txt exist in the directory.")
        return
    
    print("Resume loaded successfully!")
    print("Job description loaded successfully!")
    print()
    
    # Calculate match score
    match_score = calculate_resume_match(resume_content, job_desc_content)
    
    # Print result
    print(f"Resume Match Score: {match_score}%")
    
    # Provide interpretation
    if match_score > 70:
        print("✅ Excellent match! Strong candidate.")
    elif match_score > 50:
        print("✅ Good match! Potential candidate.")
    else:
        print("⚠️  Low match. May not be suitable.")

if __name__ == "__main__":
    main()

