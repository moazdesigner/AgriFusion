import unittest
import os
from src.inference import SoilHealthPredictor

class TestInference(unittest.TestCase):
    def setUp(self):
        # Setup a dummy model path (will trigger random weights warning, which is fine for logic testing)
        self.predictor = SoilHealthPredictor(model_path="dummy_path.pth")
        
        # Create a dummy image
        from PIL import Image
        self.dummy_img_path = "test_img.jpg"
        Image.new('RGB', (100, 100), color='red').save(self.dummy_img_path)

    def test_prediction_output_structure(self):
        text = "Soil is dry and red."
        result = self.predictor.predict(text, self.dummy_img_path)
        
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)

    def tearDown(self):
        if os.path.exists(self.dummy_img_path):
            os.remove(self.dummy_img_path)

if __name__ == '__main__':
    unittest.main()