import cv2
from transformers import CLIPProcessor, CLIPModel
from gtts import gTTS
from flask import Flask, request, send_file

# Load and preprocess images
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Generate image descriptions
def generate_description(image):
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    inputs = processor(text=["a description for my image"], images=image, return_tensors="pt", padding=True)
    logits_per_image = model(**inputs).logits_per_image
    probs = logits_per_image.softmax(dim=-1)
    description = processor.decode(probs.argmax(dim=-1)[0])

    print(f"Generated description: {description}")  # Add this line

    return description

# Convert text to speech
def text_to_speech(text, output_filename):
    tts = gTTS(text)
    tts.save(output_filename)

# Write the description to a text file
def write_to_text_file(text, filename):
    with open(filename, 'w') as f:
        f.write(text)

# Set up a Flask server
app = Flask(__name__)

@app.route('/image-to-audio', methods=['POST'])
def image_to_audio():
    file = request.files['image']
    file.save('input_image.jpg')
    
    image = load_image('input_image.jpg')
    description = generate_description(image)
    print(f"Generated description: {description}")  # Debug print
    
    # Check if the description is valid
    if description and not description.isspace() and not description == "!":
        # Write the description to a text file
        write_to_text_file(description, 'output_text.txt')
        print(f"Type of description: {type(description)}")
        text_to_speech(description, 'output_audio.mp3')
        return send_file('output_audio.mp3', as_attachment=True)
    else:
        return "No valid description generated", 500


if __name__ == '__main__':
    app.run()
