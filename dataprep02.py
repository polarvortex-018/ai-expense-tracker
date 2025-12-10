import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.model_selection import train_test_split
import json
import os

# --- PATH CONFIG ---
# Assuming 'processed_data' folder was created by dataprep01.py in the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, 'processed_data')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_training_data.csv')
MODEL_NAME = 'expense_category_model.tflite'
TOKENIZER_CONFIG_NAME = 'tokenizer_config.json'

# --- HYPERPARAMETERS ---
VOCAB_SIZE = 5000     # Max number of unique words to keep
MAX_LENGTH = 20       # Max length of input sentence
EMBEDDING_DIM = 64    # Size of the word vector space
EPOCHS = 30           # Number of training cycles
# ---------------------

def train_and_convert_model():
    print(f"Loading data from {PROCESSED_DATA_FILE}...")
    try:
        df = pd.read_csv(PROCESSED_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_FILE}. Run dataprep01.py first.")
        return

    # 1. Prepare Data for Model
    sentences = df['cleaned_text'].tolist()
    labels = df['category_id'].values
    NUM_CATEGORIES = len(np.unique(labels))

    # --- 2. TOKENIZATION (Converts words to numerical sequences needed by the model) ---
    print("\nFitting tokenizer and converting text to sequences...")
    
    # oov_token is used for words not seen during training
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # Save the tokenizer configuration for the Flutter app
    tokenizer_config = {
        'word_index': tokenizer.word_index,
        'vocab_size': VOCAB_SIZE,
        'max_length': MAX_LENGTH,
        'oov_token_id': tokenizer.word_index.get(tokenizer.oov_token)
    }
    
    config_filepath = os.path.join(PROCESSED_DATA_DIR, TOKENIZER_CONFIG_NAME)
    with open(config_filepath, 'w') as f:
        json.dump(tokenizer_config, f, indent=4)
    print(f"Saved tokenizer configuration to {config_filepath}")

    # --- 3. TRAIN/TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    
    # --- 4. MODEL DEFINITION ---
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        GlobalAveragePooling1D(), 
        Dense(24, activation='relu'),
        Dense(NUM_CATEGORIES, activation='softmax') # Softmax outputs probabilities for each category
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # --- 5. MODEL TRAINING ---
    print("\nStarting Model Training...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1 
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Model Training Complete ---")
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    
    # --- 6. TENSORFLOW LITE CONVERSION (The Deployment Step) ---
    print("\nConverting model to TFLite format for mobile deployment...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization reduces model size significantly
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    tflite_model = converter.convert()

    # Save the TFLite model file
    tflite_filepath = os.path.join(PROCESSED_DATA_DIR, MODEL_NAME)
    with open(tflite_filepath, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model successfully created and saved to: {tflite_filepath}")
    print("\n--- Python ML Phase Complete ---")

if __name__ == '__main__':
    train_and_convert_model()