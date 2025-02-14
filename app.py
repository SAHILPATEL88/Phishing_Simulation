# from flask import Flask, render_template, request, jsonify
# import pickle  # Assuming you're using a pickled ML model

# app = Flask(__name__)

# # Load the ML model (ensure the file exists in the same directory)
# model = pickle.load(open('model.pkl', 'rb'))
# vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# # Route for dashboard
# @app.route('/')
# def dashboard():
#     return render_template('home.html')

# @app.route('/home')
# def home_classifier():
#     return render_template('home.html')


# # Route for the Spam Classifier page
# @app.route('/spam')
# def spam_classifier():
#     return render_template('spam.html')

# # Route for predicting Spam/Not Spam
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         input_sms = data.get('input_sms')

#         # Preprocess the input and make predictions
#         if input_sms:
#             transformed_input = vectorizer.transform([input_sms])
#             prediction = model.predict(transformed_input)

#             # Convert numerical prediction to readable text
#             result = "Spam" if prediction[0] == 1 else "Not Spam"
#             return jsonify({"prediction": result})
#         else:
#             return jsonify({"error": "No input provided"}), 400

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
import os
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for,send_file
from flask_cors import CORS
import sqlite3
import numpy as np
import pandas as pd
import requests
import pickle
import random
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load the model
loaded_model = pickle.load(open('model_phis.pkl', 'rb'))

# Verify the loaded model (optional)
print(type(loaded_model)) 

try:
    feature_names = loaded_model.feature_names_in_
    print("Feature names used during model training:")
    print(feature_names)
except AttributeError:
    print("Model does not contain feature names. Ensure that the model was trained with feature names in a DataFrame.")

import numpy as np

class FeatureExtraction:
    def __init__(self, url):
        self.url = url

    def getFeaturesList(self):
        # Extract features based on the names from your model
        features = [
            self.usingIP(),  # 'UsingIP'
            self.longURL(),  # 'LongURL'
            self.shortURL(),  # 'ShortURL'
            self.symbolAt(),  # 'Symbol@'
            self.redirectingSlash(),  # 'Redirecting//'
            self.prefixSuffixDash(),  # 'PrefixSuffix-'
            self.subDomains(),  # 'SubDomains'
            self.https(),  # 'HTTPS'
            self.domainRegLen(),  # 'DomainRegLen'
            self.favicon(),  # 'Favicon'
            self.nonStdPort(),  # 'NonStdPort'
            self.httpsDomainURL(),  # 'HTTPSDomainURL'
            self.requestURL(),  # 'RequestURL'
            self.anchorURL(),  # 'AnchorURL'
            self.linksInScriptTags(),  # 'LinksInScriptTags'
            self.serverFormHandler(),  # 'ServerFormHandler'
            self.infoEmail(),  # 'InfoEmail'
            self.abnormalURL(),  # 'AbnormalURL'
            self.websiteForwarding(),  # 'WebsiteForwarding'
            self.statusBarCust(),  # 'StatusBarCust'
            self.disableRightClick(),  # 'DisableRightClick'
            self.usingPopupWindow(),  # 'UsingPopupWindow'
            self.iframeRedirection(),  # 'IframeRedirection'
            self.ageOfDomain(),  # 'AgeofDomain'
            self.dnsRecording(),  # 'DNSRecording'
            self.websiteTraffic(),  # 'WebsiteTraffic'
            self.pageRank(),  # 'PageRank'
            self.googleIndex(),  # 'GoogleIndex'
            self.linksPointingToPage(),  # 'LinksPointingToPage'
            self.statsReport()  # 'StatsReport'
        ]
        
        print("Extracted Features: ", features)  # Debugging step to print extracted features
        return features

    def usingIP(self):
        # Example logic for 'UsingIP' feature
        return 1 if 'http' in self.url else 0  # Modify this with actual logic

    def longURL(self):
        # Example logic for 'LongURL' feature
        return 1 if len(self.url) > 50 else 0

    def shortURL(self):
        # Example logic for 'ShortURL' feature
        return 1 if len(self.url) < 20 else 0

    def symbolAt(self):
        return self.url.count('@')

    def redirectingSlash(self):
        return self.url.count('//')

    def prefixSuffixDash(self):
        return 1 if '-' in self.url else 0

    def subDomains(self):
        return self.url.count('.')

    def https(self):
        return 1 if self.url.startswith('https') else 0

    def domainRegLen(self):
        # Add logic for domain registration length
        return len(self.url.split('.')[0])

    def favicon(self):
        # Example logic for favicon feature
        return 1 if 'favicon' in self.url else 0

    def nonStdPort(self):
        # Example logic for non-standard port
        return 1 if ':' in self.url else 0

    def httpsDomainURL(self):
        return 1 if 'https' in self.url else 0

    def requestURL(self):
        # Example logic for RequestURL
        return 1 if 'request' in self.url else 0

    def anchorURL(self):
        return 1 if '<a href=' in self.url else 0

    def linksInScriptTags(self):
        return 1 if 'script' in self.url else 0

    def serverFormHandler(self):
        return 1 if 'form' in self.url else 0

    def infoEmail(self):
        return 1 if 'email' in self.url else 0

    def abnormalURL(self):
        return 1 if 'abnormal' in self.url else 0

    def websiteForwarding(self):
        return 1 if 'forwarding' in self.url else 0

    def statusBarCust(self):
        return 1 if 'status' in self.url else 0

    def disableRightClick(self):
        return 1 if 'right-click' in self.url else 0

    def usingPopupWindow(self):
        return 1 if 'popup' in self.url else 0

    def iframeRedirection(self):
        return 1 if 'iframe' in self.url else 0

    def ageOfDomain(self):
        return 1 if len(self.url.split('.')[0]) > 3 else 0

    def dnsRecording(self):
        return 1 if 'dns' in self.url else 0

    def websiteTraffic(self):
        return 1 if 'traffic' in self.url else 0

    def pageRank(self):
        return 1 if 'pagerank' in self.url else 0

    def googleIndex(self):
        return 1 if 'index' in self.url else 0

    def linksPointingToPage(self):
        return 1 if 'links' in self.url else 0

    def statsReport(self):
        return 1 if 'stats' in self.url else 0

    
API_KEY = '7e30b3bb2f00e431d9bc09371023c0e4bb9bcea4cc644b19a23125717613905d'  # Replace with your VirusTotal API key
VIRUSTOTAL_SCAN_URL = "https://www.virustotal.com/api/v3/files"
VIRUSTOTAL_ANALYSIS_URL = "https://www.virustotal.com/api/v3/analyses/"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Database initialization for login form
def init_db():
    if not os.path.exists('database.db'):
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

# Initialize database
init_db()

# Spam Classifier Components
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    model = None
    vectorizer = None
    print("Model or vectorizer file not found.")

@app.route('/vir')
def home():
    return render_template('virus.html')

@app.route('/')
def index():
    return render_template('index.html')  # Serve your HTML file



@app.route('/scan-file', methods=['POST'])
def scan_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    headers = {
        "x-apikey": API_KEY
    }

    # Send file to VirusTotal API for scanning
    response = requests.post(VIRUSTOTAL_SCAN_URL, headers=headers, files={"file": file})
    if response.status_code == 200:
        scan_data = response.json()

        # Extract analysis ID from the response
        analysis_id = scan_data.get("data", {}).get("id", None)
        if not analysis_id:
            return jsonify({'error': 'No analysis ID found in response'}), 500
        
        # Perform a second request to get detailed scan information
        analysis_url = f'https://www.virustotal.com/api/v3/analyses/{analysis_id}'
        analysis_response = requests.get(analysis_url, headers=headers)

        # Log the response to verify the structure
        print("Analysis Response:", analysis_response.json())  # Debugging: Log full response

        if analysis_response.status_code == 200:
            detailed_data = analysis_response.json()

            # Extract the necessary details from the detailed data
            file_details = detailed_data.get("data", {}).get("attributes", {})
            metadata = {
                "sha256": scan_data.get("data", {}).get("id", "Unknown"),
                "scan_date": file_details.get("last_analysis_date", "Unknown Date")
            }

            details = {
                "file_name": file_details.get("file_name", "Unknown File"),
                "file_size": file_details.get("size", "Unknown Size"),
                "file_type": file_details.get("type", "Unknown Type"),
                "metadata": metadata,
                "scan_results": file_details.get("last_analysis_results", {})
            }

            return jsonify({
                "status": "File scanned successfully!",
                "scan_id": metadata["sha256"],  # This should be the SHA256 hash of the file
                "details": details
            })
        else:
            return jsonify({"error": "Failed to fetch detailed analysis data"}), 500
    else:
        return jsonify({"error": response.json().get("error", {}).get("message", "Unknown error")}), 500




       

# # Database initialization for login form
# def init_db():
#     if not os.path.exists('database.db'):
#         conn = sqlite3.connect('database.db')
#         cursor = conn.cursor()
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 email TEXT NOT NULL,
#                 password TEXT NOT NULL
#             )
#         ''')
#         conn.commit()
#         conn.close()

# # Initialize database
# init_db()

# # Spam Classifier Components
# try:
#     model = pickle.load(open('model.pkl', 'rb'))
#     vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
# except FileNotFoundError:
#     model = None
#     vectorizer = None
#     print("Model or vectorizer file not found.")

# OpenPhish feed URL for phishing data
OPENPHISH_URL = 'https://openphish.com/feed.txt'

# Fetch phishing data from OpenPhish
def fetch_phishing_data():
    try:
        response = requests.get(OPENPHISH_URL)
        response.raise_for_status()  # Ensure the response is valid
        phishing_urls = response.text.splitlines()  # List of URLs
        return phishing_urls
    except requests.RequestException as e:
        print(f"Error fetching phishing data: {e}")
        return []

# Generate random latitude and longitude
def generate_random_coordinates():
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon


# Routes for Dashboard
@app.route('/')
def dashboard():
    return render_template('home.html')

@app.route('/home')
def home_classifier():
    return render_template('home.html')

# Routes for Login Form




def init_db():
    with sqlite3.connect('database1.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

# Initialize the database
init_db()

@app.route('/')
def login():
    # Serve the login page
    return render_template('ac1.html','ac2.html','ac3.html','ac4.html')
    

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Retrieve email and password from the form
        email = request.form['email']
        password = request.form['password']
        print(f"Received Data - Email: {email}, Password: {password}")

        # Save the data in the database
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
            conn.commit()

        # Redirect to the home page with a success message
        return redirect(url_for('login', message="Login successful!"))
    
    except sqlite3.IntegrityError:
        # Handle duplicate email entries
        return redirect(url_for('login', message="Email already exists. Please try again."))

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404 - Page Not Found</h1>", 404




# Routes for Spam Classifier
@app.route('/spam')
def spam_classifier():
    return render_template('spam.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_sms = data.get('input_sms')

        if input_sms and model and vectorizer:
            transformed_input = vectorizer.transform([input_sms])
            prediction = model.predict(transformed_input)
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            return jsonify({"prediction": result})
        else:
            return jsonify({"error": "No input provided or model unavailable"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Routes for Live Phishing Attack Map
@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/get_phishing_data')
def get_phishing_data():
    phishing_urls = fetch_phishing_data()
    coordinates = []

    for url in phishing_urls[:10]:  # Limit to first 10 for now
        lat, lon = generate_random_coordinates()
        coordinates.append({
            'url': url,
            'latitude': lat,
            'longitude': lon
        })

    return jsonify(coordinates)

# Proxy route for external API (handle CORS issues)
@app.route('/proxy', methods=['GET'])
def proxy():
    external_url = "https://extensions.chatgptextension.ai/extensions/app/get_key"
    try:
        response = requests.get(external_url)
        response.raise_for_status()  # Ensure the response is valid (status code 2xx)

        # Forward the JSON response to the client
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500

# Adding after_request to ensure CORS headers are set properly
@app.after_request
def after_request(response):
    if request.path == '/get_phishing_data':
        response.headers['Access-Control-Allow-Origin'] = '*'
    return response
@app.route("/phish", methods=["GET", "POST"])
def phish_classifier():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)  # Using the FeatureExtraction class to get features
        features = obj.getFeaturesList()  # List of features
        
        # Convert the feature list to a numpy array with the correct shape
        x = np.array(features).reshape(1, -1)  # Reshape to 2D array for prediction

        # Use the loaded model to make predictions
        y_pred = loaded_model.predict(x)[0]
        y_pro_phishing = loaded_model.predict_proba(x)[0, 0]
        y_pro_non_phishing = loaded_model.predict_proba(x)[0, 1]

        # Prediction result
        pred = f"It is {y_pro_phishing*100:.2f} % safe to go"
        return render_template('phishing.html', xx=round(y_pro_non_phishing, 2), url=url)

    return render_template("phishing.html", xx=-1)



if __name__ == '__main__':
    print("CORS is enabled")  # Print confirmation that CORS is enabled
    app.run(debug=True)
