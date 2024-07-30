import os
import cv2
import numpy as np
import pandas as pd
from datetime import date, datetime

# Initialize recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('attSysClassifier.xml')

# Load the face cascade
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

# Prepare the list of names
path = [os.path.join("imgs/", file) for file in os.listdir("imgs/")]
names = []
for image in path:
    name = image.split(".")[0]
    name = name.split("/")[1]
    names.append(name)

names = list(set(names))
print(names)

# Open the webcam
cam = cv2.VideoCapture(0)

# Create or reset the attendance file
file_path = "attendance.xlsx"
if not os.path.exists(file_path):
    df_old = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    df_old.to_excel(writer, index=False, sheet_name='attendance')
    writer.save()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.9,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less than 100 ==> "0": perfect match
        if confidence < 100:
            id = names[id - 1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )

        if id != "unknown":
            time_now = datetime.now()
            current_time = time_now.strftime("%H:%M:%S")

            # Try to read the existing file
            try:
                df_old = pd.read_excel(file_path, engine="openpyxl")
            except Exception as e:
                print(f"Error reading '{file_path}': {e}")
                df_old = pd.DataFrame(columns=['Name', 'Date', 'Time'])

            dataframe = pd.DataFrame({
                'Name': [str(id)],
                'Date': [str(date.today())],
                'Time': [str(current_time)]
            })

            # Append new data to the DataFrame
            df_old = pd.concat([df_old, dataframe], ignore_index=True)

            # Save the updated DataFrame to Excel
            try:
                writer = pd.ExcelWriter(file_path, engine='openpyxl')
                df_old.to_excel(writer, index=False, sheet_name='attendance')
                writer.save()
            except Exception as e:
                print(f"Error saving '{file_path}': {e}")

    cv2.imshow('camera', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
