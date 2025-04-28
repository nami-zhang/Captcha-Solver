import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Set directories: adjust these paths if needed
CAPTCHA_DIR = "./saved_captchas"         # Folder where auto-generated CAPTCHA images are stored
LABELED_DIR = "./labeled_captchas"         # Folder where labeled images will be saved

# Create the labeled folder if it doesn't exist
if not os.path.exists(LABELED_DIR):
    os.makedirs(LABELED_DIR, exist_ok=True)

# Set model directory and device
MODEL_DIR = "./trocr-finetuned-captcha"  # Path to your finetuned model
DEVICE = "cuda"

print("Loading model...")
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()  # Set model to evaluation mode

def predict(image, top_n=1):
    """
    Runs model inference on the given image and returns a list of predictions.
    Confidence scores are not used.
    """
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        output_sequences = model.generate(
            pixel_values,
            max_length=10,  # Adjust based on your expected CAPTCHA length
            num_return_sequences=top_n,
            num_beams=top_n,
            return_dict_in_generate=True
        )
    predictions = [processor.tokenizer.decode(seq, skip_special_tokens=True)
                   for seq in output_sequences.sequences]
    return predictions

def sanitize_filename(name):
    """
    Removes any non-alphanumeric characters from the predicted name for safe filenames.
    """
    return "".join(c for c in name if c.isalnum())

def main():
    # List all PNG images in the CAPTCHA_DIR
    images = [f for f in os.listdir(CAPTCHA_DIR) if f.lower().endswith('.png')]
    print(f"Found {len(images)} CAPTCHA images in '{CAPTCHA_DIR}'.")
    
    for img_file in images:
        img_path = os.path.join(CAPTCHA_DIR, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_file} due to error: {e}")
            continue
        
        # Get the best prediction (top_n=1)
        predictions = predict(image, top_n=1)
        if predictions:
            best_prediction = predictions[0].strip()
        else:
            best_prediction = "unknown"
        
        # Sanitize prediction to create a safe filename
        label = sanitize_filename(best_prediction)
        if not label:
            label = "unknown"
        
        # Create new filename, ensuring uniqueness if the file already exists
        new_filename = f"{label}.png"
        new_path = os.path.join(LABELED_DIR, new_filename)
        counter = 1
        while os.path.exists(new_path):
            new_filename = f"{label}_{counter}.png"
            new_path = os.path.join(LABELED_DIR, new_filename)
            counter += 1
        
        # Save the image to the new folder with the new filename
        image.save(new_path)
        print(f"Labeled '{img_file}' as '{new_filename}'")
    
if __name__ == "__main__":
    main()
