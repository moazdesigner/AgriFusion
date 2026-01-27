import argparse
from src.inference import SoilHealthPredictor

def main():
    parser = argparse.ArgumentParser(description="Predict Soil Nutrient Deficiency")
    parser.add_argument("--text", type=str, required=True, help="Farmer's description of the soil")
    parser.add_argument("--image", type=str, required=True, help="Path to the soil image file")
    parser.add_argument("--model", type=str, default="best_agri_model.pth", help="Path to saved model")
    
    args = parser.parse_args()
    
    # Initialize Predictor
    predictor = SoilHealthPredictor(model_path=args.model)
    
    # Run Prediction
    print("\nAnalyzing...")
    result = predictor.predict(args.text, args.image)
    
    print("-" * 30)
    print(f"Prediction: {result.get('label', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")
    print("-" * 30)

if __name__ == "__main__":
    main()