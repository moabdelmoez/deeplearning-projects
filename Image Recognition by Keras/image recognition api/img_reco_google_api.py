from base64 import b64encode

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials

IMAGE_FILE = "temple.jpg"
CREDENTIALS_FILE = "YOUR-CREDENTIAL-FILE.json"

# Connect to the Google Cloud-ML Service
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE) #read the credential file into an object
service = googleapiclient.discovery.build('vision', 'v1', credentials=credentials) #create an instance of the google api client

# Read file and convert it to a base64 encoding
with open(IMAGE_FILE, "rb") as f:
    image_data = f.read()
    encoded_image_data = b64encode(image_data).decode('UTF-8')

# Create the request object for the Google Vision API
batch_request = [{
    'image': {
        'content': encoded_image_data
    },
    'features': [
        {
            'type': 'LABEL_DETECTION'
        }
    ]
}]
request = service.images().annotate(body={'requests': batch_request})

# Send the request to Google
response = request.execute()

# Check for errors
if 'error' in response:
    raise RuntimeError(response['error'])

#print(response)

#print the results
labels = response['responses'][0]['labelAnnotations']

#print each label in a new line
for label in labels:
    print(label["description"], label["score"])
