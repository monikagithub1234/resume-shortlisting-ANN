import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")
df = df[['Resume', 'Category']].dropna()

# Encode labels
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = tfidf_vectorizer.fit_transform(df['Resume']).toarray()
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Dense(512, activation='relu', input_shape=(3000,)),
    Dense(256, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Save model and encoders
model.save("resume_ann_model.h5")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

