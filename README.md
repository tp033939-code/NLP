# Fake Review Detection System

A machine learning-powered web application for detecting computer-generated fake product reviews using classical NLP techniques and supervised learning.

## 🚀 Try It Now!

**Live Demo:** https://xd8gwrubpw5rrmyvnfri3s.streamlit.app/

Test the system instantly with the deployed application - no installation required

## 📂 Source Code

**GitHub Repository:** https://github.com/tp033939-code/NLP

View the complete source code, including the Jupyter notebook, training pipeline, and deployment files.

## 💻 Run Locally

To run the application on your local machine:

```bash
# Navigate to the project directory
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Performance Metrics (After Data Cleaning & Optimized Preprocessing)
- **Test Accuracy:** 90.73%
- **Precision:** 90.35%
- **Recall:** 91.19%
- **F1-Score:** 90.77%
- **ROC-AUC:** 96.97%

## Dataset
- **Original Size:** 40,432 reviews
- **After Duplicate Removal:** 40,412 reviews (20 duplicates removed)
- **Class Balance:** 50/50 (Fake/Genuine)
- **Categories:** 10 product types
- **Split:** 70/15/15 (Train/Validation/Test)


## Features
- Real-time fake review detection
- 90%+ accuracy using classical machine learning
- Text-based classification (no metadata required)
- Pre-loaded example reviews for testing
- Detailed probability breakdown for predictions
- Linguistic feature extraction based on classical NLP techniques
