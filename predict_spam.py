import pickle
import re
import string
from xgboost import XGBClassifier

def preprocess_text(text):
    if not text:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def check_spam_indicators(text):
    # Common spam phrases and patterns
    spam_indicators = {
        'urgent_words': ['urgent', 'immediate', 'act now', 'limited time'],
        'money_words': ['cash', 'money', 'dollars', '$', '€', '£', 'price', 'offer', 'discount'],
        'pressure_words': ['limited offer', 'exclusive deal', 'only today', 'dont miss'],
        'suspicious_words': ['verify account', 'bank details', 'winner', 'won', 'lottery', 'inheritance'],
        'action_words': ['click here', 'sign up now', 'subscribe now', 'buy now', 'order now']
    }
    
    found_indicators = []
    text = text.lower()
    
    # Count matches for each category
    matches = {category: 0 for category in spam_indicators}
    for category, words in spam_indicators.items():
        for word in words:
            if word in text:
                matches[category] += 1
    
    return matches

def load_model_and_vectorizer():
    try:
        # Load the trained model
        with open('spam_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Load the vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
            
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please make sure 'spam_model.pkl' and 'tfidf_vectorizer.pkl' exist.")
        return None, None

def predict_spam():
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        return
    
    print("\n=== Spam Email Predictor ===")
    print("Enter the email details below:")
    
    # Get user input
    subject = input("\nEnter email subject: ").strip()
    print("\nEnter email body (press Enter twice to finish):")
    
    # Collect body lines until user enters an empty line
    body_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        body_lines.append(line)
    
    body = " ".join(body_lines)
    
    # Skip if content is too short
    if len(subject.split()) + len(body.split()) < 3:
        print("\n=== Prediction Results ===")
        print("Classification: HAM (Not Spam)")
        print("Reason: Content too short for reliable analysis")
        return
    
    # Combine and preprocess the text
    combined_text = subject + " " + body
    processed_text = preprocess_text(combined_text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction with adjusted threshold
    probability = model.predict_proba(text_vector)[0][1]
    
    # Check for spam indicators
    indicator_matches = check_spam_indicators(combined_text)
    
    # Calculate base threshold and adjustments
    base_threshold = 0.7  # Higher base threshold
    
    # Count total matches across categories
    total_matches = sum(indicator_matches.values())
    
    # Adjust threshold based on indicators (more indicators = lower threshold needed)
    threshold_adjustment = min(0.2, total_matches * 0.05)  # Cap the adjustment
    final_threshold = max(0.5, base_threshold - threshold_adjustment)
    
    is_spam = probability >= final_threshold
    
    # Generate detailed analysis
    print("\n=== Prediction Results ===")
    print(f"Classification: {'SPAM' if is_spam else 'HAM (Not Spam)'}")
    print(f"Confidence: {probability:.2%}")
    
    # Show confidence level
    if abs(probability - 0.5) > 0.4:
        confidence_level = "High"
    elif abs(probability - 0.5) > 0.2:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    print(f"Confidence Level: {confidence_level}")
    
    # If classified as spam, show the reasons
    if is_spam:
        print("\nSpam Indicators Found:")
        for category, count in indicator_matches.items():
            if count > 0:
                category_name = category.replace('_', ' ').title()
                print(f"- {category_name}: {count} match(es)")
        
        print("\nRecommendation: This email shows characteristics commonly associated with spam.")
    else:
        print("\nNo significant spam indicators found.")
    
    # Show threshold information for transparency
    print(f"\nAnalysis Details:")
    print(f"Base Threshold: {base_threshold:.2f}")
    print(f"Final Threshold: {final_threshold:.2f}")
    print(f"Spam Probability: {probability:.2f}")

if __name__ == "__main__":
    while True:
        predict_spam()
        
        # Ask if user wants to check another email
        print("\nWould you like to check another email? (yes/no)")
        if input().lower().strip() != 'yes':
            print("\nThank you for using the Spam Predictor!")
            break 