# AI Resume Screening System

An interactive AI-powered resume screening application built using Python, Streamlit, and Scikit-learn.  
The system compares a candidate's resume with a job description and calculates a match score using Natural Language Processing techniques.

This project simulates the basic functionality of an Applicant Tracking System (ATS) used by recruiters.

------------------------------------------------------------

FEATURES

• Interactive web interface built with Streamlit  
• Paste resume text and job description for analysis  
• TF-IDF vectorization for text processing  
• Cosine similarity for match score calculation  
• Real-time match score display  
• Match classification (Excellent, Good, Moderate, Low)  
• Simple and responsive user interface  

------------------------------------------------------------

TECH STACK

Python  
Streamlit  
Scikit-learn  
NumPy  
Natural Language Processing (TF-IDF, Cosine Similarity)

------------------------------------------------------------

PROJECT STRUCTURE

AI-RESUME-SCREENING/

app.py                → Streamlit web application  
resume_screening.py   → Command-line version  
resume.txt            → Sample resume text  
job_description.txt   → Sample job description  
requirements.txt      → Project dependencies  
README.md             → Project documentation  

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install -r requirements.txt

------------------------------------------------------------

RUN THE APPLICATION

Start the Streamlit web application:

streamlit run app.py

After running the command, open the browser and go to:

http://localhost:8501

------------------------------------------------------------

HOW IT WORKS

1. The system reads the resume and job description text.  
2. Text data is converted into numerical vectors using TF-IDF Vectorizer.  
3. Cosine similarity calculates similarity between the vectors.  
4. The similarity value is converted into a percentage match score.  
5. The system categorizes the match based on the score.

------------------------------------------------------------

MATCH SCORE INTERPRETATION

> 70%        → Excellent Match  
50% – 70%    → Good Match  
30% – 50%    → Moderate Match  
< 30%        → Low Match  

------------------------------------------------------------

EXAMPLE OUTPUT

Resume Match Score: 78.45%

Excellent Match! Strong candidate.

------------------------------------------------------------

FUTURE IMPROVEMENTS

• Upload and analyze PDF resumes  
• Highlight matching keywords  
• Batch resume screening for multiple candidates  
• Export screening results to CSV  
• Integrate advanced NLP models (spaCy / BERT)

------------------------------------------------------------

AUTHOR

Anjali Kankanala

------------------------------------------------------------

AI + Streamlit = Smarter Resume Screening

