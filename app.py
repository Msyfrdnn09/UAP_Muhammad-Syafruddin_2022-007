import streamlit as st
import pandas as pd
from PIL import Image
from utils.image_predictor import ImagePredictor

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Garbage Classification Dashboard",
    layout="wide"
)

st.title("🗑️ Klasifikasi Jenis Sampah Daur Ulang")
st.write("Upload gambar sampah dan lihat hasil prediksi menggunakan model Deep Learning.")

# ===============================
# Sidebar - Model Selection
# ===============================
st.sidebar.title("⚙️ Pengaturan Model")

# 🔑 MODEL MAP (FORMAT .keras — WAJIB)
model_map = {
    "MobileNetV2 (Rekomendasi)": "mobilenet_model.keras",
    "CNN Base": "cnn_model.keras",
    "ResNet50": "resnet50_model.keras"
}

model_name = st.sidebar.selectbox(
    "Pilih Model",
    list(model_map.keys())
)

MODEL_PATH  = f"models/{model_map[model_name]}"
CONFIG_PATH = "models/preprocess_config.json"
LABELS_PATH = "models/labels.txt"

# ===============================
# Load Predictor (Inference Only)
# ===============================
predictor = ImagePredictor(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    labels_path=LABELS_PATH
)

# ===============================
# Upload Image
# ===============================
uploaded_file = st.file_uploader(
    "Upload gambar (.jpg / .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Gambar Input",
            use_column_width=True
        )

    with col2:
        result = predictor.predict(image)

        st.subheader("📌 Hasil Prediksi")
        st.success(f"Kelas: **{result['label']}**")
        st.write(f"Confidence: **{result['confidence'] * 100:.2f}%**")

        st.subheader("📊 Probabilitas Tiap Kelas")
        prob_df = (
            pd.DataFrame.from_dict(
                result["all_predictions"],
                orient="index",
                columns=["Probabilitas"]
            )
            .sort_values("Probabilitas", ascending=False)
        )

        st.bar_chart(prob_df)

# ===============================
# Model Comparison Table
# ===============================
st.divider()
st.subheader("📈 Perbandingan Model")

df = pd.read_csv("reports/model_comparison.csv")
st.dataframe(df, use_container_width=True)

st.caption("Model terbaik: MobileNetV2 (berdasarkan Accuracy & F1-score)")
