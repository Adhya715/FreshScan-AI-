import streamlit as st
from pathlib import Path
import tempfile
from src.inference.predict import predict

st.title("FreshScan AI")

fruit_emojis = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Strawberry": "🍓"
}

st.write("This app helps you classify a fruit as Fresh or Rotten by simply uploading its picture.")

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

if st.button("🔍 Predict"):

    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        fruit, condition, confidence = predict(temp_path)

        st.subheader("Prediction Result")

        st.write(f"{fruit_emojis.get(fruit, '🍇')} **Fruit:** {fruit}")

        if condition == "Fresh":
            st.success("✅ Fresh")
        else:
            st.error("❌ Rotten")

        st.progress(confidence, text=f"{confidence:.2%} confidence")

        st.write(f"Confidence: **{confidence:.2%}**")