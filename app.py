import os
import numpy as np
from flask import Flask, request, render_template
import joblib
import PyPDF2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model and tools
model = load_model("resume_ann_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'resume_file' not in request.files:
        return render_template("index.html", prediction=None)
    
    file = request.files['resume_file']
    
    if file.filename == '':
        return render_template("index.html", prediction=None)

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text
        text = ''
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if not text.strip():
            return render_template("index.html", prediction=[("Could not extract text from PDF.", "")])

        x_input = vectorizer.transform([text])
        predictions = model.predict(x_input)[0]

        # Get top 3 predictions
        top_indices = predictions.argsort()[-3:][::-1]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_scores = [round(predictions[i] * 100, 2) for i in top_indices]

        top_predictions = list(zip(top_labels, top_scores))
        return render_template("index.html", prediction=top_predictions)

    return render_template("index.html", prediction=[("Invalid file format. Please upload a PDF.", "")])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
