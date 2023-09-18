from PIL import Image
import numpy as np
import urllib3
from io import BytesIO
import json

# Pre-processing function
def preProcess(data):
    # Read the PNG image received from the post request
    image = Image.open(data)

    # Resize the image to 28x28 using bilinear interpolation
    resized_image = image.resize((28, 28), Image.BILINEAR)

    # Convert the image to a NumPy array
    image_array = np.array(resized_image)

    # Calculate grayscale manually
    gray_array = np.mean(image_array, axis=-1, keepdims=True).astype(np.uint8)
    gray_array = gray_array.reshape(28, 28, 1)

    # Finally change the reshape
    return gray_array.reshape((1, 28, 28, 1))

def lambda_handler(event, context):
    # Create an HTTP connection pool
    http = urllib3.PoolManager()

    # Send an HTTP GET request to the specified URL
    response = http.request('GET', event['img_url'])

    # Check for HTTP errors
    if response.status != 200:
        return {
            'statusCode': response.status,
            'body': 'HTTP request failed'
        }

    # Do pre-processing
    result_array = preProcess(BytesIO(response.data))

    # Send the processed image back to the UI
    return {
        'statusCode': 200,
        'body': json.dumps({'Pre Processed Image': result_array.tolist()})
    }
