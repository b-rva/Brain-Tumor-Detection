import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import datetime

# Load model
model = tf.keras.models.load_model("G:/My Drive/Brain Tumor Detection/brain_tumor_finetuned.h5")

# Define class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Create directory for flagged images
flagged_dir = "flagged_images"
os.makedirs(flagged_dir, exist_ok=True)

# Prediction function
def predict_image(img):
    img = img.resize((299, 299))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probabilities = {class_names[i]: float(predictions[0][i]) for i in range(4)}
    return probabilities

# Flagging function to save images
def flag_image(img, prediction):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(flagged_dir, f"flagged_{timestamp}.png")
    img.save(img_path)

    # Save prediction result
    pred_text = f"Image: {img_path}\nPrediction: {prediction}\n\n"
    with open(os.path.join(flagged_dir, "flagged_predictions.txt"), "a") as f:
        f.write(pred_text)

    return f"✅ Flagged and saved: {img_path}"

# Project Info Page Content
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

# Custom JavaScript to load local favicon
favicon_script = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    var link = document.createElement('link');
    link.rel = 'icon';
    link.type = 'image/x-icon';
    link.href = 'favicon.ico';  // Ensure this file is in the same directory as your script
    document.head.appendChild(link);
});
</script>
"""

# Create Gradio UI with Tabs
with gr.Blocks() as demo:

    demo.title = "Brain Tumor Detection"

    # Inject JavaScript to change favicon
    demo.load(lambda: None, inputs=[], outputs=[], js=favicon_script)

    gr.Markdown("<h1 style='text-align: center;'>Brain Tumor Detection</h1>")

    with gr.Tabs():
        # Prediction Page
        with gr.Tab("🔍 Tumor Detection"):
            gr.Markdown("Upload an MRI image to detect brain tumors.")

            with gr.Row():
                # Left Side: Image Upload + Submit & Clear Buttons
                with gr.Column(scale=1):
                    # img = gr.Image(type="pil", label="Upload MRI Image", sources=["upload", "webcam"])
                    img = gr.Image(type="pil", label="Upload MRI Image", sources=["upload", "clipboard", "webcam"])
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                # Right Side: Prediction Output + Flag Button
                with gr.Column(scale=1):
                    output = gr.Label(label="Prediction")
                    flag_btn = gr.Button("Flag", variant="secondary")

            # Full-width Flag Status Below
            flag_status = gr.Textbox(label="Flag Status", interactive=False)

            # Button actions
            submit_btn.click(fn=predict_image, inputs=img, outputs=output)
            flag_btn.click(fn=flag_image, inputs=[img, output], outputs=flag_status)
            clear_btn.click(fn=lambda: (None, "", ""), inputs=[], outputs=[img, output, flag_status])

        # Project Info Page
        with gr.Tab("📖 Project Info"):
            gr.Markdown(project_description)

# Run in Notebook
demo.launch(debug=True)  # Debug mode to prevent auto-close
