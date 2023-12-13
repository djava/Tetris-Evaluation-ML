import requests
import json
import random

# Generate 10 random integers between 0 and 10
random_heights = [random.randint(0, 10) for _ in range(10)]

# Construct the request body
data = {
    "heights": [random_heights, random_heights]
}

print(data)

# URL to make the POST request
url = "https://leo-the-lion-1102097bdb43.herokuapp.com/multi-predict"
# url = "http://localhost:8000/predict"


# Make the POST request with JSON data
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    print("Response from server:", response.json())
else:
    print("Failed to get a response from the server, status code:", response.status_code)
