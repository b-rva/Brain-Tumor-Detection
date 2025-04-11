import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import os
import datetime
import requests

# ------------------ Model Download from Hugging Face ------------------
model_url = "https://huggingface.co/spaces/b-rva/brain-tumor-detection/blob/main/brain_tumor_finetuned.h5"
model_path = "brain_tumor_finetuned.h5"

# Download the model if not already present
if not os.path.exists(model_path):
    print("🔽 Downloading model from Hugging Face...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("✅ Download complete.")

# Load the model
model = tf.keras.models.load_model(model_path)

# ------------------ Class Info & Flagging Setup ------------------
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Directory for flagged images
flagged_dir = "flagged_images"
os.makedirs(flagged_dir, exist_ok=True)

# ------------------ Prediction & Flagging Functions ------------------
def predict_image(img):
    img = img.resize((299, 299))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    probabilities = {class_names[i]: float(predictions[0][i]) for i in range(4)}
    return probabilities

def flag_image(img, prediction):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(flagged_dir, f"flagged_{timestamp}.png")
    img.save(img_path)

    pred_text = f"Image: {img_path}\nPrediction: {prediction}\n\n"
    with open(os.path.join(flagged_dir, "flagged_predictions.txt"), "a") as f:
        f.write(pred_text)

    return f"✅ Flagged and saved: {img_path}"

# ------------------ Project Info ------------------
project_description = """
# 🧠 Brain Tumor Detection Project

## 📌 Project Overview
- **Objective:** Classify brain tumors using deep learning.
- **Model Used:** Xception (Transfer Learning)
- **Frameworks:** TensorFlow, Keras
- **Frontend UI:** Gradio
- **Deployment:** Cloud-based hosting (Hugging Face)

## 📌 Dataset
- **Source:** Kaggle
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Image Size:** 299x299
- **Total Images:** 7023

## 📌 Model Training
- **Optimizer:** Adam (LR: 0.0001)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, Precision, Recall
- **Training Steps:** Initial Training → Fine-Tuning

## 📌 Evaluation
- **Test Accuracy:** 96%
- **Confusion Matrix & Classification Report Used**
- **Misclassification Analysis Done**

## 📌 UI & Deployment
- **Gradio used for interactive UI**
- **Flag system for incorrect predictions**
- **Deployment on Hugging Face**
"""

# ------------------ Favicon JavaScript ------------------
favicon_script = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    var link = document.createElement('link');
    link.rel = 'icon';
    link.type = 'image/x-icon';
    link.href = 'favicon.ico';
    document.head.appendChild(link);
});
</script>
"""

# ------------------ Gradio UI ------------------
with gr.Blocks() as demo:

    demo.title = "Brain Tumor Detection"

    # Inject JS
    demo.load(lambda: None, inputs=[], outputs=[], js=favicon_script)

    gr.Markdown("<h1 style='text-align: center;'>Brain Tumor Detection</h1>")

    with gr.Tabs():
        # Tab 1: Detection
        with gr.Tab("🔍 Tumor Detection"):
            gr.Markdown("Upload an MRI image to detect brain tumors.")
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="pil", label="Upload MRI Image", sources=["upload", "clipboard", "webcam"])
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                with gr.Column(scale=1):
                    output = gr.Label(label="Prediction")
                    flag_btn = gr.Button("Flag", variant="secondary")

            flag_status = gr.Textbox(label="Flag Status", interactive=False)

            submit_btn.click(fn=predict_image, inputs=img, outputs=output)
            flag_btn.click(fn=flag_image, inputs=[img, output], outputs=flag_status)
            clear_btn.click(fn=lambda: (None, "", ""), inputs=[], outputs=[img, output, flag_status])

        # Tab 2: Project Info
        with gr.Tab("📖 Project Info"):
            gr.Markdown(project_description)

# ------------------ Launch ------------------
demo.launch(debug=True)
