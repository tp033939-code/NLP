import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data once"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        # Don't use st.error in cached function - return error instead
        return False

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define model paths relative to script location
    model_path = os.path.join(script_dir, 'models', 'final_model.pkl')
    vectorizer_path = os.path.join(script_dir, 'models', 'tfidf_vectorizer.pkl')
    results_path = os.path.join(script_dir, 'models', 'final_results.pkl')

    # Load model
    if not os.path.exists(model_path):
        return None, None, None, f"Model file not found: {model_path}"

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load vectorizer
    if not os.path.exists(vectorizer_path):
        return None, None, None, f"Vectorizer file not found: {vectorizer_path}"

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Load results (optional)
    results = None
    if os.path.exists(results_path):
        try:
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        except Exception:
            pass  # Results are optional, ignore errors

    return model, vectorizer, results, None

# Initialize preprocessing tools
@st.cache_resource
def init_preprocessing():
    """Initialize preprocessing tools"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stopwords_to_keep = {'not', 'no', 'nor', 'very', 'too', 'so', 'more', 'most'}
    stop_words = stop_words - stopwords_to_keep
    return lemmatizer, stop_words

def preprocess_text(text, lemmatizer, stop_words):
    """
    Preprocess review text (identical to training pipeline)
    
    Args:
        text (str): Raw review text
        lemmatizer: WordNet lemmatizer
        stop_words (set): Stopwords to remove
        
    Returns:
        str: Preprocessed text
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    # Join tokens back into string
    return ' '.join(tokens)

def predict_review(text, model, vectorizer, lemmatizer, stop_words):
    """
    Predict if a review is fake or genuine
    
    Args:
        text (str): Review text
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        lemmatizer: WordNet lemmatizer
        stop_words (set): Stopwords set
        
    Returns:
        tuple: (prediction, confidence, probabilities_dict)
    """
    # Preprocess
    processed_text = preprocess_text(text, lemmatizer, stop_words)
    
    # Transform to TF-IDF
    text_tfidf = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Get class order from model
    classes = model.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Get confidence for the predicted class
    predicted_idx = class_to_idx[prediction]
    confidence = probabilities[predicted_idx]
    
    # Create probability dictionary with consistent ordering
    prob_dict = {
        'CG': probabilities[class_to_idx['CG']],
        'OR': probabilities[class_to_idx['OR']]
    }
    
    return prediction, confidence, prob_dict

# Page configuration
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Download NLTK data
    nltk_success = download_nltk_data()
    if not nltk_success:
        st.error("Failed to download NLTK data. Please check your internet connection.")
        return

    # Load model and preprocessing tools
    model, vectorizer, results, error = load_model_and_vectorizer()

    if error:
        st.error(error)
        return

    if model is None or vectorizer is None:
        st.error("Failed to load model or vectorizer. Please check if model files exist.")
        return

    lemmatizer, stop_words = init_preprocessing()
    
    # Header
    st.title("🔍 Fake Review Detection System")
    st.markdown("### Detect computer-generated fake reviews using Machine Learning")
    st.markdown("**Test Accuracy: 90.11%** | **ROC-AUC: 96.60%** | Model: Logistic Regression (C=10.0, L2 penalty)")
    st.markdown("---")
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Performance")
        
        if results:
            st.markdown(f"**Model:** {results.get('model_name', 'Logistic Regression')}")
            
            # Best parameters
            best_params = results.get('best_params', {})
            if best_params:
                st.markdown("**Best Parameters:**")
                st.code(best_params)
            
            # Test set performance
            st.markdown("### Test Set Metrics")
            test_metrics = results.get('test_metrics', {})
            if test_metrics:
                st.metric("Accuracy", f"{test_metrics.get('accuracy', 0)*100:.2f}%")
                st.metric("Precision", f"{test_metrics.get('precision', 0)*100:.2f}%")
                st.metric("Recall", f"{test_metrics.get('recall', 0)*100:.2f}%")
                st.metric("F1-Score", f"{test_metrics.get('f1', 0)*100:.2f}%")
                st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0)*100:.2f}%")
        else:
            st.markdown("**Model:** Logistic Regression")
            st.markdown("**Hyperparameters:** C=10.0, L2 penalty")
            st.markdown("### Test Set Metrics")
            st.metric("Accuracy", "90.11%")
            st.metric("Precision", "90.00%")
            st.metric("Recall", "90.24%")
            st.metric("F1-Score", "90.12%")
            st.metric("ROC-AUC", "96.60%")
        
        st.markdown("---")
        st.markdown("### Dataset Info")
        st.markdown("""
        - **Size:** 40,432 reviews
        - **Balance:** 50/50 (CG/OR)
        - **Categories:** 10 products
        - **Split:** 70/15/15 (train/val/test)
        """)

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses **Logistic Regression** with
        **TF-IDF features (n-grams 1-3)** to detect
        fake reviews based on linguistic patterns.
        """)
    
    # Main content - Input section
    st.header("📝 Enter a Review to Analyze")
    
    # Create tabs
    tab1, tab2 = st.tabs(["✍️ Custom Review", "📋 Example Reviews"])
    
    with tab1:
        # Text input
        review_text = st.text_area(
            "Paste or type a product review below:",
            height=150,
            placeholder="e.g., This product is amazing! I love it so much. Highly recommend to everyone!",
            help="Enter any product review text to check if it's likely to be fake or genuine."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        # Prediction
        if analyze_button and review_text:
            if len(review_text.strip()) < 10:
                st.warning("Please enter a longer review (at least 10 characters)")
            else:
                with st.spinner("Analyzing review..."):
                    try:
                        prediction, confidence, probabilities = predict_review(
                            review_text, model, vectorizer, lemmatizer, stop_words
                        )
                        
                        # Results section
                        st.markdown("---")
                        st.header("📊 Analysis Results")
                        
                        # Main prediction
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prediction == 'CG':
                                st.error("### 🚨 FAKE REVIEW DETECTED")
                                st.markdown("This review appears to be **computer-generated (fake)**")
                                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                            else:
                                st.success("### ✅ GENUINE REVIEW")
                                st.markdown("This review appears to be **original (genuine)**")
                                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                        
                        with col2:
                            # Probability breakdown
                            st.markdown("### 📈 Probability Breakdown")
                            
                            fake_prob = probabilities['CG'] * 100
                            genuine_prob = probabilities['OR'] * 100
                            
                            st.markdown(f"**Fake (CG):** {fake_prob:.1f}%")
                            st.progress(fake_prob / 100)
                            
                            st.markdown(f"**Genuine (OR):** {genuine_prob:.1f}%")
                            st.progress(genuine_prob / 100)
                        
                        # Interpretation
                        st.markdown("---")
                        st.markdown("### 💡 Interpretation")
                        
                        if confidence >= 0.90:
                            st.info("**Very High Confidence:** The model is very confident in this prediction.")
                        elif confidence >= 0.75:
                            st.info("**High Confidence:** The model is confident in this prediction.")
                        elif confidence >= 0.60:
                            st.warning("**Moderate Confidence:** The prediction is somewhat uncertain.")
                        else:
                            st.warning("**Low Confidence:** The review has mixed characteristics.")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        
        elif analyze_button:
            st.warning("Please enter a review to analyze")
    
    with tab2:
        st.markdown("### 📋 Try These Example Reviews")
        
        # Initialize session state for example results
        if 'example_results' not in st.session_state:
            st.session_state.example_results = {}
        
        examples = {
            "Example 1: Fake Review": 
                "Probably best movie Rutger Howerton has made. Not too many people make a movie like this. "
                "If you want to watch a movie with Rutger Howerton, you will like this one. It is a very good movie.",
            
            "Example 2: Genuine Review":
                "I haven't read much Stephen King but I really enjoyed the Shining. It has been referenced so much "
                "in pop culture and has influenced so many works it was great to finally read it.",
            
            "Example 3: Fake Review":
                "great backpack lots of space and a nice quality.Very good quality.Nice bag.Very pretty.",
            
            "Example 4: Genuine Review":
                "Perfect fit, great price. Revived the helmet to functional use. Beats buying a new helmet when "
                "the helmet is in excellent condition."
        }
        
        for title, text in examples.items():
            with st.expander(title):
                st.write(text)
                if st.button(f"Test this review", key=title):
                    try:
                        prediction, confidence, probabilities = predict_review(
                            text, model, vectorizer, lemmatizer, stop_words
                        )
                        
                        # Store results in session state
                        result_type = "FAKE" if prediction == 'CG' else "GENUINE"
                        st.session_state.example_results[title] = {
                            'prediction': result_type,
                            'confidence': confidence,
                            'probabilities': probabilities
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                # Display results if they exist in session state
                if title in st.session_state.example_results:
                    result = st.session_state.example_results[title]
                    result_type = result['prediction']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    color = "red" if result_type == 'FAKE' else "green"
                    
                    st.markdown(f"**Prediction:** :{color}[{result_type}]")
                    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                    st.markdown(f"Fake: {probabilities['CG']*100:.1f}% | Genuine: {probabilities['OR']*100:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem 0;'>
            <p><strong>Fake Review Detection System</strong> | NLP Project 2025</p>
            <p>Built with Streamlit • scikit-learn • NLTK • TF-IDF</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
