import os
import torch
import random
import string
import base64
import io
from torch import Tensor
from PIL import Image
from captcha.image import ImageCaptcha
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from flask import Flask, jsonify, request
from flask_cors import CORS

# Initialize the Flask app once
app = Flask(__name__)
CORS(app)

# Set paths and device
MODEL_DIR = "./trocr-finetuned-captcha"  # Path to your finetuned model
DEVICE = "cuda"  # Change to "cpu" if you are not using a GPU

print("Loading model...")
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()  # Set model to evaluation mode

def predict_with_confidence(image, top_n=3):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        output_sequences = model.generate(
            pixel_values,
            max_length=10,  # Adjust based on your expected CAPTCHA length
            num_return_sequences=top_n,
            num_beams=top_n,
            return_dict_in_generate=True,
            output_scores=True
        )
    predictions = [processor.tokenizer.decode(seq, skip_special_tokens=True)
                   for seq in output_sequences.sequences]
    confidences = [torch.exp(log_prob).item() for log_prob in output_sequences.sequences_scores]
    return list(zip(predictions, confidences))

@app.route('/solve', methods=['POST'])
def solve_captcha():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image"}), 400

    # Run model inference
    results = predict_with_confidence(image, top_n=3)

    # Convert image to base64 so it can be sent in JSON
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = {
        "captcha_image": img_str,
        "predictions": [
            {"prediction": pred, "confidence": conf} for pred, conf in results
        ]
    }
    return jsonify(response)

if __name__ == '__main__':
    # Bind to all interfaces for AWS deployment
    app.run(host='0.0.0.0', debug=True)
