# resume-shortlisting-ANN
# Resume Classification & Shortlisting using ANN 🧠📄

This project is a machine learning web app that classifies resumes into various job categories (like Data Science, HR, etc.) and shortlists candidates based on the predicted label. It uses an Artificial Neural Network (ANN) model built with TensorFlow and deployed using Flask.

## 🚀 Features

- Upload resumes in **PDF format**
- **Extracts text** from resumes using PyPDF2
- Classifies into categories using a **trained ANN model**
- Shows **top 3 predictions with confidence scores**
- Web interface built with **HTML + CSS**
- Outputs the **shortlisted category**

## 📂 Project Structure

resume_shortlist_ann_project/
├── app.py # Flask backend
├── train_model.py # Model training script
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── label_encoder.pkl # Label encoder
├── resume_ann_model.h5 # Trained ANN model
├── UpdatedResumeDataSet.csv # Resume dataset
├── /uploads # Uploaded PDF resumes
└── /templates
└── index.html # Frontend HTML page


## 🛠️ Technologies Used

- Python 🐍
- TensorFlow/Keras 🤖
- scikit-learn 📊
- Flask 🌐
- PyPDF2 📄
- HTML/CSS 🎨

## 📊 Dataset

Used the public dataset `UpdatedResumeDataSet.csv` containing resumes and their labeled job categories.

## 🧠 Model Info

- ANN with 3 layers:
  - Dense(512) → ReLU
  - Dense(256) → ReLU
  - Dense(output) → Softmax
- Trained with `sparse_categorical_crossentropy` loss and `adam` optimizer.
- TF-IDF used for feature extraction (top 3000 words).

## 🔧 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/resume-shortlist-ann.git
   cd resume-shortlist-ann
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
Open in browser:
Visit http://127.0.0.1:5000 and upload a PDF resume.

📬 Contact
Name – NAGAM MONIKA PRIYA
LinkedIn - https://www.linkedin.com/in/monika-priya-nagam/

