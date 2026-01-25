import pandas as pd
import numpy as np
import os
import random


# Path to CSV file
CSV_PATH = 'Fertilizer Prediction.csv'

# Path to image dataset folder
# The script expects folders: 'Black Soil', 'Cinder Soil', 'Laterite Soil', 'Yellow Soil', 'Peat Soil'
IMAGE_DIR = 'Soil types' 

# Output file name
OUTPUT_CSV = 'final_soil_data.csv'

# Mapping CSV Soil Types to Image Folder Names
# Strategy: Map similar visual characteristics
SOIL_MAPPING = {
    'Black': 'Black Soil',      # Direct match
    'Red': 'Laterite Soil',     # Laterite is typically red/rusty
    'Sandy': 'Yellow Soil',     # Sandy soils can be yellowish
    'Clayey': 'Peat Soil',      # Peat is heavy/organic, closest proxy to Clayey in this set
    'Loamy': 'Cinder Soil'      # Loamy is a mix; Cinder is the remaining texture-heavy class
}

def generate_farmer_description(row):
    """
    Generates a realistic farmer description based on numerical data.
    """
    # 1. Moisture Description
    if row['Moisture'] < 30:
        moisture_text = "The soil is very dry and dusty."
    elif row['Moisture'] < 60:
        moisture_text = "The soil has moderate moisture."
    else:
        moisture_text = "The soil is damp and holds water well."

    # 2. Color/Texture Description (Based on Soil Type)
    color_text = f"It looks like {row['Soil Type']} soil."
    
    # 3. Crop Context
    crop_text = f"We are trying to grow {row['Crop Type']}."

    # 4. Nutrient Hints (Simulating visual symptoms)
    symptom_text = ""
    if row['Nitrogen'] < 13: # Low N
        symptom_text = "The leaves are turning yellow."
    elif row['Phosphorous'] < 10: # Low P
        symptom_text = "The roots seem weak and growth is stunted."
    elif row['Potassium'] < 4: # Low K
        symptom_text = "The leaf edges are scorched and brown."
    else:
        symptom_text = "The plants look relatively healthy."

    # Combine into a natural sentence
    full_text = f"{moisture_text} {color_text} {crop_text} {symptom_text}"
    return full_text

def determine_deficiency(row):
    """
    Determines the primary nutrient deficiency label.
    Thresholds based on dataset quantiles (approx 25th percentile).
    """
    # Thresholds
    N_thresh, P_thresh, K_thresh = 13, 10, 4
    
    deficiencies = []
    if row['Nitrogen'] < N_thresh: deficiencies.append('N_Deficient')
    if row['Phosphorous'] < P_thresh: deficiencies.append('P_Deficient')
    if row['Potassium'] < K_thresh: deficiencies.append('K_Deficient')
    
    if not deficiencies:
        return 'Healthy'
    else:
        # Return the first found deficiency 
        return deficiencies[0]

def get_random_image(soil_type, base_dir, mapping):
    """
    Selects a random image from the corresponding folder.
    """
    folder_name = mapping.get(soil_type)
    if not folder_name:
        return None
    
    folder_path = os.path.join(base_dir, folder_name)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        # Fallback for testing if images aren't present yet
        return f"{folder_name}/dummy_image.jpg"
        
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if images:
        selected_img = random.choice(images)
        return os.path.join(folder_path, selected_img)
    else:
        return f"{folder_name}/missing_image.jpg"


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    print("Generating descriptions and labels...")
    # Generate Descriptions
    df['text_description'] = df.apply(generate_farmer_description, axis=1)
    
    # Generate Labels
    df['deficiency_label'] = df.apply(determine_deficiency, axis=1)

    print("Assigning images...")
    # Assign Images
    df['image_path'] = df['Soil Type'].apply(lambda x: get_random_image(x, IMAGE_DIR, SOIL_MAPPING))

    # Select final columns
    final_df = df[['text_description', 'image_path', 'deficiency_label', 'Soil Type', 'Nitrogen', 'Phosphorous', 'Potassium']]
    
    print(f"Saving final dataset to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Done! Here is a preview:")
    print(final_df.head())

if __name__ == "__main__":
    main()