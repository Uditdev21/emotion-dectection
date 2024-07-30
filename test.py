import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os

# Path to the image
image_path = "D:\\codes\\pyprojects\\emo\\projectcode\\images (1).jpeg"

# Check if the file exists
if not os.path.isfile(image_path):
    print(f"File not found: {image_path}")
else:
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Failed to load image: {image_path}")
    else:
        # Analyze the image for emotions
        try:
            analysis = DeepFace.analyze(image, actions=['emotion'])
            
            # Debug print to inspect the analysis response
            print("Analysis Result:", analysis)

            # Get the dominant emotion and face region
            emotion = analysis['dominant_emotion']
            region = analysis['region']

            # Draw a rectangle around the face and label it
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the resulting image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error analyzing image: {e}")
