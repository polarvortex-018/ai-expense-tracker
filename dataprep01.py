import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import json
import os

# --- CONFIGURE THESE LINES BASED ON YOUR DATA ---
# 1. Path to your source CSV file (relative to this script or absolute)
# Update this path to point to your actual expenses.csv file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'expenses.csv')  # Place expenses.csv in the same directory
# 2. Name of the column containing the purchase description/text input
TEXT_COLUMN = 'name'                         
# 3. Name of the column containing the high-level category (the label)
CATEGORY_COLUMN = 'category'                 
# 4. Folder where output files will be saved
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'processed_data')
# ---------------------------------------------


def clean_text_for_model(text):
    """
    Cleans text by converting to lowercase and removing punctuation/extra spaces.
    (Simple cleaning since amounts are in a separate column).
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    
    # Remove punctuation 
    text = re.sub(r'[^\w\s]', '', text) 
    
    # Standardize spaces 
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_data():
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Starting data preparation from {DATA_FILE}...")
    try:
        # Load data
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}. Please check DATA_FILE path.")
        return

    # 1. Clean the text inputs (the feature the ML model will learn from)
    df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_text_for_model)
    
    # 2. Drop any rows where cleaning resulted in an empty description
    df = df[df['cleaned_text'] != '']
    
    print(f"Successfully cleaned data. Total entries: {len(df)}")
    
    # 3. Encode Category Labels (Converting text labels to numerical IDs)
    label_encoder = LabelEncoder()
    # categories are the labels the ML model will output (0, 1, 2, ...)
    df['category_id'] = label_encoder.fit_transform(df[CATEGORY_COLUMN])
    
    # 4. Save the Category MAPPING (CRUCIAL for Flutter)
    # This maps the numerical ID back to the human-readable category name
    # Format: {"Food & Dining": 0, "Transport": 1, ...}
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    
    # Save mapping to a JSON file
    map_filepath = os.path.join(OUTPUT_DIR, 'category_map.json')
    with open(map_filepath, 'w') as f:
        # We invert the dictionary for easier use in Python: {0: "Food & Dining", ...}
        inverted_mapping = {v: k for k, v in label_mapping.items()}
        json.dump(inverted_mapping, f, indent=4)
        print(f"\nSaved Category ID Map (Total {len(inverted_mapping)} Categories): {map_filepath}")

    # 5. Save the processed data ready for Model Training
    data_for_training = df[['cleaned_text', 'category_id', CATEGORY_COLUMN]].copy()
    processed_filepath = os.path.join(OUTPUT_DIR, 'processed_training_data.csv')
    data_for_training.to_csv(processed_filepath, index=False)
    print(f"Saved processed data for training: {processed_filepath}")
    
    print("\nData Preparation Complete. Ready for Model Training.")
    
    return data_for_training

if __name__ == '__main__':
    prepare_data()
