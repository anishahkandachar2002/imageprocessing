import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import cv2
import face_recognition
from io import BytesIO

# Define the URLs of the target websites
website1_url = 'http://localhost/spurce/source1/'
website2_url = 'http://localhost/spurce/source2/'

# Send GET requests to the websites
response1 = requests.get(website1_url)
response2 = requests.get(website2_url)

# Parse the HTML content of the websites
soup1 = BeautifulSoup(response1.content, 'html.parser')
soup2 = BeautifulSoup(response2.content, 'html.parser')

# Find the elements containing images and usernames on website 1
image_elements1 = soup1.find_all('img')  # Adjust this based on the HTML structure of the website
username_elements1 = soup1.find_all('span', class_='username')  # Adjust this based on the HTML structure of the website

# Find the elements containing images and usernames on website 2
image_elements2 = soup2.find_all('img')  # Adjust this based on the HTML structure of the website
username_elements2 = soup2.find_all('span', class_='username')  # Adjust this based on the HTML structure of the website

# Create a dictionary to store the image URLs and corresponding usernames with their respective sources
image_username_mapping = {}

# Extract the image URLs and usernames from website 1
for image_element, username_element in zip(image_elements1, username_elements1):
    image_url = image_element['src']  # Adjust this based on the HTML structure of the website
    username = username_element.text.strip()  # Adjust this based on the HTML structure of the website
    image_username_mapping[image_url] = {'username': username, 'source': 'website1'}

# Extract the image URLs and usernames from website 2
for image_element, username_element in zip(image_elements2, username_elements2):
    image_url = image_element['src']  # Adjust this based on the HTML structure of the website
    username = username_element.text.strip()  # Adjust this based on the HTML structure of the website
    image_username_mapping[image_url] = {'username': username, 'source': 'website2'}

# Load the uploaded image
uploaded_image_path = 'test1.jpg'
uploaded_image = Image.open(uploaded_image_path)

# Preprocess the uploaded image
uploaded_image = uploaded_image.resize((256, 256))  # Resize the image if necessary
uploaded_image = uploaded_image.convert('RGB')
uploaded_image_array = np.array(uploaded_image)

# Set the threshold for face comparison
threshold = 0.6  # Adjust this value based on your requirements

# Initialize the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to compare two faces
def compare_faces(face1, face2):
    # Convert the face images to grayscale
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_RGB2GRAY)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_RGB2GRAY)

    # Convert the grayscale images to NumPy arrays
    face1_array = np.array(face1_gray)
    face2_array = np.array(face2_gray)

    # Encode the face descriptors
    face1_encoding = face_recognition.face_encodings(face1_array)[0]
    face2_encoding = face_recognition.face_encodings(face2_array)[0]

    # Compare the face encodings
    face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)

    # Calculate the similarity score
    similarity_score = 1.0 - face_distance[0]

    return similarity_score




# Iterate over the image URLs and compare with the uploaded image
for image_url, user_info in image_username_mapping.items():
    response = requests.get(image_url)
    website_image = Image.open(BytesIO(response.content))
    website_image = website_image.resize((256, 256))  # Resize the image if necessary
    website_image = website_image.convert('RGB')
    website_image_array = np.array(website_image)

    # Detect faces in the uploaded image
    uploaded_gray = cv2.cvtColor(uploaded_image_array, cv2.COLOR_RGB2GRAY)
    uploaded_faces = face_cascade.detectMultiScale(uploaded_gray, scaleFactor=1.1, minNeighbors=5)

    # Detect faces in the website image
    website_gray = cv2.cvtColor(website_image_array, cv2.COLOR_RGB2GRAY)
    website_faces = face_cascade.detectMultiScale(website_gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate over the detected faces and compare
    for (ux, uy, uw, uh) in uploaded_faces:
        uploaded_face = uploaded_image_array[uy:uy+uh, ux:ux+uw]

        for (wx, wy, ww, wh) in website_faces:
            website_face = website_image_array[wy:wy+wh, wx:wx+ww]

            # Perform face comparison on the extracted faces
            similarity = compare_faces(uploaded_face, website_face)

            if similarity > threshold:
                print('Match found for username:', user_info['username'])
                print('Source:', user_info['source'])
