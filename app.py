from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import os
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
from feature_extr import FeatureExtraction
warnings.filterwarnings('ignore')
import email  # Add this import

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Load the trained model and vectorizer for email spam detection
def load_model_and_vectorizer():
    try:
        with open('spam_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return None, None
    

# Load phishing detection model
phishing_model = pickle.load(open("gbc_final_model.pkl", "rb"))


# Preprocess text
def preprocess_text(text):
    if not text:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Check for spam indicators
def check_spam_indicators(text):
    spam_indicators = {
        'urgent_words': ['urgent', 'immediate', 'act now', 'limited time'],
        'money_words': ['cash', 'money', 'dollars', '$', '€', '£', 'price', 'offer', 'discount'],
        'pressure_words': ['limited offer', 'exclusive deal', 'only today', 'dont miss'],
        'suspicious_words': ['verify account', 'bank details', 'winner', 'won', 'lottery', 'inheritance'],
        'action_words': ['click here', 'sign up now', 'subscribe now', 'buy now', 'order now']
    }
    
    text = text.lower()
    
    # Count matches for each category
    matches = {category: 0 for category in spam_indicators}
    for category, words in spam_indicators.items():
        for word in words:
            if word in text:
                matches[category] += 1
    
    # Generate reasons based on matches
    reasons = []
    if matches['urgent_words'] >= 1:
        reasons.append("Contains urgent or immediate action words")
    if matches['money_words'] >= 2:
        reasons.append("Contains multiple references to money or financial terms")
    if matches['pressure_words'] >= 1:
        reasons.append("Uses pressure tactics or limited-time offers")
    if matches['suspicious_words'] >= 1:
        reasons.append("Contains suspicious keywords often found in scams")
    if matches['action_words'] >= 2:
        reasons.append("Contains multiple call-to-action phrases")
    
    return matches, reasons

# Load model and vectorizer at startup
model, vectorizer = load_model_and_vectorizer()

# Routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    try:
        print("Received email analysis request")
        data = request.get_json()
        print("Request data:", data)
        body = data.get('body', '')
        print("Email body:", body)

        # Preprocess text
        processed_text = preprocess_text(body)

        
        if len(processed_text.split()) < 3:
            return jsonify({
                'title': "=== Prediction Results ===",
                'classification': "HAM (Not Spam)",
                'reason': "Content too short for reliable analysis"
            })

        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model or vectorizer not loaded'}), 500

        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])

        # Make prediction
        probability = model.predict_proba(text_vector)[0][1]
        
        # Check for spam indicators and get reasons
        indicator_matches, reasons = check_spam_indicators(body)
        
        # Calculate base threshold and adjustments
        base_threshold = 0.70
        total_matches = sum(indicator_matches.values())
        threshold_adjustment = min(0.20, total_matches * 0.05)
        final_threshold = max(0.5, base_threshold - threshold_adjustment)
        
        is_spam = probability >= final_threshold

        # Prepare response to match predict_spam.py output
        response = {
            'title': "=== Prediction Results ===",
            'classification': 'SPAM' if is_spam else 'HAM (Not Spam)',
            'spamIndicators': {
                category.replace('_', ' ').title(): count 
                for category, count in indicator_matches.items() 
                if count > 0
            } if is_spam else {},
            'reasons': reasons if is_spam else ["No significant spam indicators found."],
            'recommendation': (
                "This email shows characteristics commonly associated with spam."
                if is_spam else 
                "Email appears to be legitimate."
            ),
            'analysisDetails': {
                'baseThreshold': f"{base_threshold:.2f}",
                'finalThreshold': f"{final_threshold:.2f}",
                'spamProbability': f"{probability:.2f}"
            }
        }

        return jsonify(response)

    except Exception as e:
        print("Error in analyze_email:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_eml', methods=['POST'])
def analyze_eml():
    try:
        print("Received EML file analysis request")  # Debug print
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        if not data or 'emlContent' not in data:
            return jsonify({'error': 'No EML content provided'}), 400
            
        eml_content = data['emlContent']
        
        # Parse EML content
        msg = email.message_from_string(eml_content)
        subject = msg.get('subject', '')
        print("Extracted subject:", subject)  # Debug print
        
        # Get email body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()
        
        print("Extracted body:", body[:100])  # Debug print first 100 chars
        
        # Combine and preprocess text
        combined_text = subject + ' ' + body
        processed_text = preprocess_text(combined_text)

        # Skip empty or very short texts
        if len(processed_text.split()) < 3:
            return jsonify({
                'title': "=== Prediction Results ===",
                'classification': "HAM (Not Spam)",
                'reason': "Content too short for reliable analysis"
            })

        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model or vectorizer not loaded'}), 500

        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])

        # Make prediction
        probability = model.predict_proba(text_vector)[0][1]
        
        # Check for spam indicators and get reasons
        indicator_matches, reasons = check_spam_indicators(body)
        
        # Calculate base threshold and adjustments
        base_threshold = 0.70
        total_matches = sum(indicator_matches.values())
        threshold_adjustment = min(0.20, total_matches * 0.05)
        final_threshold = max(0.5, base_threshold - threshold_adjustment)
        
        is_spam = probability >= final_threshold

        # Prepare response
        response = {
            'title': "=== Prediction Results ===",
            'classification': 'SPAM' if is_spam else 'HAM (Not Spam)',
            'spamIndicators': {
                category.replace('_', ' ').title(): count 
                for category, count in indicator_matches.items() 
                if count > 0
            } if is_spam else {},
            'reasons': reasons if is_spam else ["No significant spam indicators found."],
            'recommendation': (
                "This email shows characteristics commonly associated with spam."
                if is_spam else 
                "Email appears to be legitimate."
            ),
            'analysisDetails': {
                'baseThreshold': f"{base_threshold:.2f}",
                'finalThreshold': f"{final_threshold:.2f}",
                'spamProbability': f"{probability:.2f}"
            }
        }
        return jsonify(response)

    except Exception as e:
        print("Error in analyze_eml:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url', '')

        # Extract features
        feature_extractor = FeatureExtraction(url)
        features = feature_extractor.getFeaturesList()

        # Ensure extracted features are not more than 30
        max_features = 30
        features = features[:max_features] + [None] * (max_features - len(features))

        # Feature names and detailed descriptions
        FEATURE_INFO = [
            ("Using IP (UsingIP)", "If the domain contains an IP address instead of a domain name, it's more likely to be phishing."),
            ("Long URL (LongURL)", "Long URLs are often used to hide malicious parameters."),
            ("Short URL (ShortURL)", "Shortened URLs can obscure the real destination and may lead to phishing websites."),
            ("Symbol '@' (Symbol@)", "The '@' symbol in a URL is often used in phishing attacks to create fake subdomains."),
            ("Redirecting with // (Redirecting//)", "URLs with multiple forward slashes can be used for redirection and deception."),
            ("Prefix-Suffix in Domain (PrefixSuffix-)", "A hyphen in the domain name is often a sign of phishing attempts."),
            ("Subdomains (SubDomains)", "Excessive subdomains can be used to mimic legitimate sites."),
            ("HTTPS (HTTPS)", "The presence of HTTPS does not guarantee safety but increases legitimacy."),
            ("Domain Registration Length (DomainRegLen)", "Short registration periods indicate a higher likelihood of phishing."),
            ("Favicon (Favicon)", "If the favicon is missing or mismatched, it might indicate phishing."),
            ("Non-Standard Port (NonStdPort)", "Phishing sites often use uncommon ports to evade detection."),
            ("HTTPS in Domain (HTTPSDomainURL)", "Having 'https' in the domain name instead of using it properly in the URL is suspicious."),
            ("Request URL (RequestURL)", "Phishing sites often load resources from external sources."),
            ("Anchor URL (AnchorURL)", "Links within the page that redirect to suspicious domains indicate phishing."),
            ("Links in Script Tags (LinksInScriptTags)", "If many external links are found in JavaScript, it could indicate phishing."),
            ("Server Form Handler (ServerFormHandler)", "If the form action points to an external domain, it is risky."),
            ("Info Email (InfoEmail)", "Email addresses in page content can indicate phishing."),
            ("Abnormal URL (AbnormalURL)", "If the URL structure deviates from standard formats, it can be suspicious."),
            ("Website Forwarding (WebsiteForwarding)", "Frequent redirections are a known phishing tactic."),
            ("Status Bar Customization (StatusBarCust)", "Altering the browser status bar is a sign of deception."),
            ("Right Click Disable (DisableRightClick)", "Disabling right-click prevents users from investigating the site."),
            ("Popup Window (UsingPopupWindow)", "Excessive pop-ups are often a phishing tactic."),
            ("Iframe Redirection (IframeRedirection)", "Hidden iframes can be used to steal information."),
            ("Age of Domain (AgeofDomain)", "Newly registered domains are more likely to be malicious."),
            ("DNS Record (DNSRecording)", "A missing DNS record suggests that a site might not be trustworthy."),
            ("Website Traffic (WebsiteTraffic)", "Low traffic websites are often malicious."),
            ("PageRank (PageRank)", "A low PageRank means the site is not well-trusted."),
            ("Google Index (GoogleIndex)", "If a site is not indexed by Google, it could be a phishing site."),
            ("Links Pointing to Page (LinksPointingToPage)", "Legitimate sites have more backlinks."),
            ("Statistical Report (StatsReport)", "Phishing sites often appear in blacklists.")
        ]

        # Create feature descriptions
        feature_descriptions = []
        for i in range(max_features):
            feature_name, full_description = FEATURE_INFO[i]
            feature_value = features[i]

            if feature_value == 1:
                meaning = "✅ Indicates the behaviour of a legitimate website."
            elif feature_value == -1:
                meaning = "⚠️ Indicates phishing behavior."
            elif feature_value == 0:
                meaning = "ℹ️ No strong indication of phishing or legitimacy."

            feature_descriptions.append({
                "feature": feature_name,
                "description": meaning,
                "full_description": full_description  # Ensuring full_description is defined properly
            })

        # Predict phishing probability
        url_features = np.array([f if f is not None else 0 for f in features]).reshape(1, -1)
        prediction = phishing_model.predict(url_features)[0]
        
        #Define response Messages
        if prediction == 1:
            classification = "Legitimate"
            conclusion = "✅ Safe to Visit: The analysis indicates that the URL does not exhibit characteristics of phishing. While no automated system is 100% accurate, this website appears to be safe for browsing. However, always exercise caution when entering sensitive information online."
        else:
            classification = "Phishing"
            conclusion = "⚠️ Caution: The URL you entered has been identified as a phishing website. Phishing websites are designed to steal sensitive information such as login credentials, credit card details, or personal data. It is strongly recommended that you do not enter any personal information on this site and avoid interacting with it."

        result = {
            "classification": "Legitimate" if prediction == 1 else "Phishing",
            "features_table": feature_descriptions,
            "conclusion": conclusion
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the application
if __name__ == '__main__':
    print("Starting Flask application...")
    print("Email Model loaded:", model is not None)
    print("Vectorizer loaded:", vectorizer is not None)
    print("Phishing Model loaded:", phishing_model is not None)
    app.run(debug=True, host='0.0.0.0', port=5000)