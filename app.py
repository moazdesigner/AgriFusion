import streamlit as st
from PIL import Image
import tempfile
import os
from src.inference import SoilHealthPredictor

# Page Config
st.set_page_config(
    page_title="AgriFusion Soil Doctor",
    page_icon="🌱",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 20px;
        background-color: #fff3cd;
        border-radius: 10px;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    # Helper to load model once and cache it
    return SoilHealthPredictor(model_path="best_agri_model.pth")

def main():
    st.title("🌱 AgriFusion: Soil Health Doctor")
    st.markdown("---")
    st.write("Upload a soil image and describe its condition to get an AI-powered nutrient analysis.")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("This Multimodal AI combines text analysis (BERT) and computer vision (ResNet) to diagnose soil nutrient deficiencies.")

    # Input Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Visual Check")
        uploaded_file = st.file_uploader("Upload Soil Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Sample", use_column_width=True)

    with col2:
        st.subheader("2. Farmer's Observation")
        text_input = st.text_area(
            "Describe the soil (Color, Texture, Moisture):", 
            height=150,
            placeholder="e.g., The soil is black and sticky. It holds water well but crops are yellowing."
        )

    # Analyze Button
    if st.button("🔍 Analyze Soil Health"):
        if uploaded_file and text_input:
            with st.spinner("Consulting the AI Agronomist..."):
                # Save temp file for the predictor
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    uploaded_file.seek(0)
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    # Load model and predict
                    predictor = load_predictor()
                    result = predictor.predict(text_input, tmp_path)
                    
                    # Display Results
                    label = result['label']
                    conf = result['confidence'] * 100
                    
                    st.markdown("### 📊 Diagnostic Results")
                    
                    if label == "Healthy":
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Status: {label}</h3>
                            <p>Confidence: <b>{conf:.1f}%</b></p>
                            <p>The soil appears to be in good condition. Continue standard maintenance.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h3>⚠️ Detected: {label}</h3>
                            <p>Confidence: <b>{conf:.1f}%</b></p>
                            <p>Action Recommended: Apply fertilizers rich in <b>{label.split('_')[0]}</b>.</p>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        else:
            st.warning("Please provide both an image and a description.")

if __name__ == "__main__":
    main()