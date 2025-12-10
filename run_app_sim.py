import tensorflow as tf
import numpy as np
import json
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

# Suppress the deprecation warning specific to interpreter deletion
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="The TFLite interpreter is being deleted,")

# --- CONFIG ---
PROCESSED_DATA_DIR = 'processed_data'
MODEL_NAME = 'expense_category_model.tflite'
TOKENIZER_CONFIG_NAME = 'tokenizer_config.json'
CATEGORY_MAP_NAME = 'category_map.json'

def load_ai_assets():
    """Loads the TFLite model, tokenizer config, and category map."""
    
    # Load Model
    model_path = os.path.join(PROCESSED_DATA_DIR, MODEL_NAME)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Load Tokenizer Config
    config_path = os.path.join(PROCESSED_DATA_DIR, TOKENIZER_CONFIG_NAME)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load Category Map (ID -> Name)
    map_path = os.path.join(PROCESSED_DATA_DIR, CATEGORY_MAP_NAME)
    with open(map_path, 'r') as f:
        category_map = {int(k): v for k, v in json.load(f).items()}
        
    return interpreter, config, category_map

def clean_text_for_model(text):
    """(Matches the logic in dataprep01.py: lowercasing and punctuation removal)"""
    # NOTE: This function must ALWAYS match the text cleaning in Dart and dataprep01.py!
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_expense(raw_input, interpreter, config, category_map):
    """Runs the full transaction simulation."""
    
    # 1. Regex Extraction (for Amount and Description)
    amount_match = re.search(r'(\d+\.?\d*)\s*(rs|inr|\$|\w*)', raw_input, re.IGNORECASE)
    
    amount = 0.0
    description = raw_input
    if amount_match:
        try:
            amount = float(amount_match.group(1))
        except ValueError:
            pass
            
        description = raw_input.replace(amount_match.group(0), '').strip()
    
    if amount == 0.0:
        return None, 0.0, False, 0.0 
        
    # 2. Tokenization and Padding
    cleaned_description = clean_text_for_model(description)
    
    tokenizer = Tokenizer()
    tokenizer.word_index = config['word_index']
    
    sequence = tokenizer.texts_to_sequences([cleaned_description])
    padded_sequence = pad_sequences(
        sequence, 
        maxlen=config['max_length'], 
        padding='post', 
        truncating='post'
    )
    
    # 3. Inference Setup and Execution
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # CRITICAL FIX: Cast the sequence to the model's expected FLOAT32 type
    input_tensor = padded_sequence.astype(np.float32) 
    
    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    
    # 4. Interpret Output
    output_data = interpreter.get_tensor(output_details['index']).flatten()
    
    predicted_id = np.argmax(output_data)
    confidence = np.max(output_data)
    predicted_category = category_map.get(predicted_id, "Unknown")
    
    # 5. Determine Income/Expense
    # Check both the predicted category AND keywords in the raw input
    # This allows income detection even though the model doesn't have income categories
    
    # Keywords that indicate income in the raw input
    income_keywords = ['salary', 'commission', 'income', 'deposit', 'refund', 
                       'bonus', 'payment received', 'freelance', 'consulting',
                       'dividend', 'interest', 'cashback', 'reimbursement']
    
    # Check if any income keyword is in the raw input (case-insensitive)
    raw_input_lower = raw_input.lower()
    keyword_match = any(keyword in raw_input_lower for keyword in income_keywords)
    
    # Income categories from the model (currently none, but kept for future use)
    income_categories = []  # Add income category names here when model is retrained
    category_match = predicted_category in income_categories
    
    # Mark as income if either keyword matches OR category matches
    is_income = keyword_match or category_match
    
    return predicted_category, amount, is_income, confidence

def main():
    try:
        interpreter, config, category_map = load_ai_assets()
    except Exception as e:
        print(f"Error loading AI assets. Ensure you have run all Python prep scripts.")
        print(f"Details: {e}")
        return

    print("\n--- Console Finance Tracker (AI Simulation) ---")
    print(f"Model Ready. Total Categories: {len(category_map)}")
    print("---------------------------------------------")
    
    # ==============================================================
    # 💰 USER INPUT SECTION: Pinpoint your 5 test values here:
    # ==============================================================
    
    user_test_cases = [
        "1200rs petrol",
        "30000rs salaray",
        "450rs taxi",
        "200rs shoes",
        "6000rs rent"
    ]
    
    # Ensure there are exactly 5 cases
    if len(user_test_cases) != 5:
        print("\n[WARNING] Please define exactly 5 test cases in the script code.")
        return

    # ==============================================================
    
    # Run tests
    for i, case in enumerate(user_test_cases):
        category, amount, is_income, confidence = predict_expense(case, interpreter, config, category_map)
        
        if amount == 0.0:
            print(f"\n[TEST {i+1} FAILED]: Could not parse amount from '{case}'.")
            continue
            
        type_label = "INCOME" if is_income else "EXPENSE"
        sign = "+" if is_income else "-"
        
        print("\n" + "="*40)
        print(f"[TEST {i+1} | {type_label}]")
        print(f"  Raw Input: {case}")
        print(f"  AMOUNT: {sign}{amount:.2f} INR")
        print(f"  AI CATEGORY: {category}")
        print(f"  Confidence: {confidence:.2f}")

if __name__ == '__main__':
    # Ensure you are running this from the ML_scripts directory after activating your environment
    main()