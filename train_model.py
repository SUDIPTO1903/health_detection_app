import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Define the model
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))  # Conv layer for feature extraction
    model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling to reduce dimensions
    model.add(Flatten())  # Flatten the 2D data into a 1D vector for the fully connected layers
    model.add(Dense(128, activation='relu'))  # Dense layer for learning high-level features
    model.add(Dense(9, activation='softmax'))  # Output layer for classification (9 emotions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load your dataset (replace with your dataset path)
def load_data(dataset_path):
    images = []
    labels = []

    # Assuming the dataset is arranged in subfolders representing emotions (e.g., 'angry', 'happy', etc.)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Stress', 'Fatigue']
    
    # The train folder inside the dataset path
    train_folder = os.path.join(dataset_path, 'train')  # Path to the 'train' folder
    
    # Check if the 'train' folder exists
    if not os.path.isdir(train_folder):
        print(f"Error: 'train' folder not found in {dataset_path}. Please check the dataset path.")
        return [], []  # Return empty lists if 'train' folder is not found
    
    for label in emotion_labels:
        emotion_folder = os.path.join(train_folder, label)
        
        # Check if the emotion folder exists
        if not os.path.isdir(emotion_folder):
            print(f"Warning: Folder {emotion_folder} not found. Skipping this folder.")
            continue  # Skip this folder if not found
        
        # List files in the folder and check the number of images
        image_files = [f for f in os.listdir(emotion_folder) if os.path.isfile(os.path.join(emotion_folder, f))]
        
        if len(image_files) == 0:
            print(f"Warning: No images found in the folder: {emotion_folder}. Skipping this folder.")
            continue  # Skip this folder if it has no images

        # Loop through image files in the folder
        for img_name in image_files:
            img_path = os.path.join(emotion_folder, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                if img is None:
                    print(f"Warning: Unable to read image {img_path}. Skipping this file.")
                    continue  # Skip invalid image files
                img = cv2.resize(img, (48, 48))  # Resize image to 48x48 pixels
                img = img_to_array(img) / 255.0  # Convert to array and normalize to [0, 1]
                images.append(img)
                labels.append(emotion_labels.index(label))  # Get the emotion label index
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue  # Skip this image if an error occurs

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Ensure data is not empty
    if len(images) == 0:
        raise ValueError("No images found in the dataset. Please check the dataset path or folder structure.")

    # Reshape images to (num_samples, 48, 48, 1) to match input shape of the model
    images = images.reshape(images.shape[0], 48, 48, 1)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=9)

    return images, labels

# Train the model
def train_model(dataset_path):
    # Load and preprocess data
    try:
        X, y = load_data(dataset_path)
    except ValueError as e:
        print(f"Error: {e}")
        return  # Exit if no data found

    # Check if data is empty after loading
    if len(X) == 0 or len(y) == 0:
        print("Error: No data loaded. Please check the dataset path and structure.")
        return
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))  # Set epochs to 2

    # Save the trained model
    model.save('expression_model.h5')
    print("Model saved as 'expression_model.h5'")

if __name__ == "__main__":
    dataset_path = r'C:/Users/HP/Downloads/archive (8)'  # Set the correct path to your dataset
    train_model(dataset_path)  # Train the model
    
    