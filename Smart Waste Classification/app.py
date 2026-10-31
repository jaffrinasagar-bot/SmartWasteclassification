
import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Classification System")
st.write("Upload a waste image to identify its category.")

st.info(
    "This project uses a pretrained AI image classification model."
)


@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="watersplash/waste-classification"
    )


def classify_category(label):
    label = label.lower().replace("_", " ").replace("-", " ")

    if "plastic" in label:
        return "Plastic Waste", "Use the appropriate plastic recycling bin."
    elif "paper" in label or "cardboard" in label:
        return "Paper Waste", "Keep paper clean and dry for recycling."
    elif "metal" in label or "can" in label:
        return "Metal Waste", "Use the appropriate metal recycling bin."
    elif "glass" in label:
        return "Glass Waste", "Use a designated glass collection bin."
    elif any(word in label for word in
             ["biological", "organic", "food", "vegetable"]):
        return "Organic Waste", "Use an approved compost or organic-waste bin."
    elif "battery" in label or "electronic" in label:
        return "E-Waste / Battery", "Use an authorized collection point."
    elif "clothes" in label or "shoes" in label:
        return "Textile Waste", "Consider reuse or textile collection."
    else:
        return "Other Waste", "Check local waste disposal guidelines."


uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Classify Waste"):
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()
                predictions = model(image, top_k=3)

            best = predictions[0]
            category, advice = classify_category(best["label"])

            st.success("Classification completed!")

            st.subheader("Prediction: " + category)
            st.write("Model label:", best["label"])
            st.write(
                f"Confidence: {best['score'] * 100:.2f}%"
            )
            st.progress(float(best["score"]))

            st.subheader("Disposal Suggestion")
            st.write(advice)

            st.subheader("Other Predictions")
            for result in predictions:
                st.write(
                    f"{result['label']}: "
                    f"{result['score'] * 100:.2f}%"
                )

            st.caption(
                "Predictions may be incorrect. Follow local recycling rules."
            )

        except Exception as error:
            st.error("Unable to classify the image.")
            st.write(str(error))
            st.info(
                "Check your internet connection and model download."
            )
else:
    st.warning("Please upload an image to get started.")

st.divider()
st.caption("Computer Vision Mini Project | Smart Waste Classification")
