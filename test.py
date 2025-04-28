import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Set paths
MODEL_DIR = "./trocr-finetuned-captcha"  # Folder where model is saved
TEST_DIR = os.path.join("dataset", "test")  # Folder with test images
DEVICE = "cuda"

# Load trained model and processor
print("Loading model...")
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()  # Set model to evaluation mode

def predict_with_confidence(image, top_n=3):
    """
    Generate multiple predictions with corresponding confidence levels.

    Args:
        image (PIL.Image): The input image.
        top_n (int): Number of top predictions to generate.

    Returns:
        List of tuples containing predictions and their confidence scores.
    """
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        output_sequences = model.generate(
            pixel_values,
            max_length=10,              # Adjust based on CAPTCHA text length
            num_return_sequences=top_n, # Get multiple outputs
            num_beams=top_n,            # Use beam search for better results
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode generated sequences to text.
    predictions = [
        processor.tokenizer.decode(seq, skip_special_tokens=True)
        for seq in output_sequences.sequences
    ]

    # Convert log-probabilities to probabilities as confidence scores.
    confidences = [torch.exp(log_prob).item() for log_prob in output_sequences.sequences_scores]
    return list(zip(predictions, confidences))

def test_model(top_n=3):
    """
    Test the model on all images in the test folder and compute overall accuracies.

    Accuracy metrics computed:
      - Top Prediction Correct (case-sensitive)
      - Any Prediction Correct (case-sensitive)
      - Top Prediction Correct (case-insensitive)
      - Any Prediction Correct (case-insensitive)
    """
    print("\nTesting model with all files from test folder...\n")
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]
    
    if not test_files:
        print("No test files found in the test folder.")
        return
    
    total_files = len(test_files)
    correct_top1 = 0                    # Case-sensitive top prediction match counter
    correct_any = 0                     # Case-sensitive any prediction match counter
    correct_top1_case_insensitive = 0   # Case-insensitive top prediction match counter
    correct_any_case_insensitive = 0    # Case-insensitive any prediction match counter
    
    for i, file_name in enumerate(test_files):
        file_path = os.path.join(TEST_DIR, file_name)
        try:
            captcha_image = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {file_name}: {e}")
            continue

        # Expected CAPTCHA text derived from the filename (without .png)
        expected = os.path.splitext(file_name)[0]
        expected_lower = expected.lower()
        predictions_confidences = predict_with_confidence(captcha_image, top_n)

        # Determine the prediction with the highest confidence score.
        top_prediction, top_confidence = max(predictions_confidences, key=lambda x: x[1])
        
        # Case-sensitive comparisons
        if top_prediction == expected:
            correct_top1 += 1
        if any(prediction == expected for prediction, _ in predictions_confidences):
            correct_any += 1

        # Case-insensitive comparisons
        if top_prediction.lower() == expected_lower:
            correct_top1_case_insensitive += 1
        if any(prediction.lower() == expected_lower for prediction, _ in predictions_confidences):
            correct_any_case_insensitive += 1

        # Print detailed results for each image.
        """
        print(f"\n[{i + 1}/{total_files}] CAPTCHA:")
        print(f"   Expected: {expected}")
        for j, (prediction, confidence) in enumerate(predictions_confidences):
            print(f"   {j + 1}. {prediction} (Confidence: {confidence:.2%})")
        """
    
    # Calculate and print overall accuracy metrics
    top1_accuracy = (correct_top1 / total_files) * 100
    any_accuracy = (correct_any / total_files) * 100
    top1_case_accuracy = (correct_top1_case_insensitive / total_files) * 100
    any_case_accuracy = (correct_any_case_insensitive / total_files) * 100
    
    print("\nOverall Accuracy:")
    print(f"   Top Prediction Correct (case-sensitive): {top1_accuracy:.2f}%")
    print(f"   Top Prediction Correct (case-insensitive): {top1_case_accuracy:.2f}%")
    print(f"   At Least One Prediction Correct (case-sensitive): {any_accuracy:.2f}%")
    print(f"   At Least One Prediction Correct (case-insensitive): {any_case_accuracy:.2f}%")

if __name__ == "__main__":
    test_model(top_n=3)
