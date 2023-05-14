
import cv2
from transformers import CLIPProcessor, CLIPModel
from gtts import gTTS
from flask import Flask, request, send_file
import requests

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

    inputs = processor(images=image, return_tensors="pt", padding=True)
    logits = model(**inputs).logits_per_image
    probs = logits.softmax(dim=-1)
    description = processor.decode(probs.argmax(dim=-1)[0])

    return description

# Convert text to speech
def text_to_speech(text, output_filename):
    tts = gTTS(text)
    tts.save(output_filename)

# Set up a Flask server
app = Flask(__name__)

@app.route('/image-to-audio', methods=['POST'])
def image_to_audio():
    file = request.files['image']
    file.save('input_image.jpg')
    
    image = load_image('input_image.jpg')
    description = generate_description(image)
    text_to_speech(description, 'output_audio.mp3')
    
    return send_file('output_audio.mp3', as_attachment=True)

if __name__ == '__main__':
    app.run()
