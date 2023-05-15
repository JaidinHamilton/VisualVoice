import requests

url = 'http://localhost:5000/image-to-audio'
image_file = '263-2638193_baby-pet-white-tiger.jpg'

with open(image_file, 'rb') as f:
    files = {'image': f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open('output_audio.mp3', 'wb') as f:
        f.write(response.content)
        print('Audio file saved as output_audio.mp3')
else:
    print('Error:', response.status_code, response.text)
