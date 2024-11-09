import cv2
import numpy as np

# Function to detect face and eyes
def detect_face_and_eyes(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade classifiers for face and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    return faces, gray_frame

# Function to analyze dark circles under eyes
def analyze_dark_circles(frame, eyes):
    for (ex, ey, ew, eh) in eyes:
        # Define regions for dark circle analysis (below the eyes)
        if ey + eh + 10 < frame.shape[0]:  # Ensure we don't go out of bounds
            dark_circle_region = frame[ey + eh:ey + eh + 10, ex:ex + ew]
            adjacent_region = frame[ey - 10:ey + eh - 10, ex:ex + ew]

            # Calculate average brightness of regions
            avg_dark_circle_intensity = np.mean(dark_circle_region)
            avg_adjacent_intensity = np.mean(adjacent_region)

            # Threshold for dark circles detection (you may need to adjust this)
            if avg_dark_circle_intensity < avg_adjacent_intensity - 15:
                return True
        
    return False

# Main function to run the webcam feed and detect sleep deprivation
def main():
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break
        
        faces, gray_frame = detect_face_and_eyes(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
            
            roi_gray = gray_frame[y:y + h, x:x + w]
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(roi_gray)

            is_sleep_deprived = analyze_dark_circles(frame[y:y+h], eyes)

            if is_sleep_deprived:
                label_text = "Sleep Deprived"
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red color for warning
            else:
                label_text = "Not Sleep Deprived"
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green color for normal

        cv2.imshow('Sleep Deprivation Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()