"""
DEPLOYMENT TO RASPBERRY PI VIA ROBOFLOW
"""
import os
from roboflow import Roboflow

rf = Roboflow(api_key="48owXBimC5K7mCyJkRgg")
project = rf.workspace("wildfire-xpwrf").project("wildfire-4tdl8")
model = project.version(1, local="http://localhost:9001/").model

# Specify the directory containing your images
image_dir = "./images"

# Get a list of all files in the directory
image_files = os.listdir(image_dir)

# Loop over all files
for image_file in image_files:
    # Construct the full file path
    image_path = os.path.join(image_dir, image_file)
    # Only predict if the file is an image (ends with .jpeg, .jpg, .png, etc.)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        prediction = model.predict(image_path, confidence=40, overlap=30)
        print(prediction.json())