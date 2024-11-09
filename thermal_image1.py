import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load a pre-trained model (e.g., MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for fever detection
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (fever or no fever)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocessing function for the thermal images
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to the size expected by the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if thermal images are in grayscale
    image = img_to_array(image)  # Convert to an array
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess for MobileNetV2
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to simulate a thermal-like effect using a colormap
def simulate_thermal_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    thermal_image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # Apply a thermal colormap
    return thermal_image

# Function to detect faces using OpenCV's Haar Cascade
def detect_face(frame):
    # Load pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

# Example of predicting fever in a video stream
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    faces = detect_face(frame)
    
    # If faces are detected, apply thermal effect only on faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for the face
            roi = frame[y:y+h, x:x+w]
            
            # Simulate thermal-like effect only on the face
            thermal_roi = simulate_thermal_image(roi)
            
            # Place the processed ROI back into the original frame
            frame[y:y+h, x:x+w] = thermal_roi
            
            # Preprocess the "thermal" frame for prediction (only for the region of interest)
            image = preprocess_image(thermal_roi)
            
            # Predict fever status
            prediction = model.predict(image)
            
            if prediction >= 0.5:
                label = "No Fever Detected"
                color = (0, 0, 255)  # Red for fever
            else:
                label = "Fever"
                color = (255, 0, 0)  # Green for no fever
            
            # Draw label on the frame
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        # If no faces are detected, show a message
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame with the result
    cv2.imshow('Fever Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
