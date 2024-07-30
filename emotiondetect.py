#run python -m venv ./venv
#run venv/scripts/activate
#pip install deepface opencv-python-headless 

import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'])

        print("Analysis Result:", analysis)

        for result in analysis:
            emotion = result['dominant_emotion']
            region = result['region']

            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)

    except Exception as e:
        print(f"Error analyzing frame: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
