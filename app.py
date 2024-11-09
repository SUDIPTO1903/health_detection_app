from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import threading
import subprocess  # To run external scripts
import os

app = Flask(__name__)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Function to generate frames from the webcam for live video feed
def generate_frames():
    while True:
        success, frame = video_capture.read()  # Read a frame from the webcam
        if not success:
            break
        
        # Display frame in JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page with a button for fever prediction

# Route to stream video from the webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve favicon.ico
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Function to start fever prediction by running thermal_image1.py
@app.route('/predict_fever', methods=['POST'])
@app.route('/predict_fever', methods=['POST'])
def predict_fever_route():
    try:
        # Execute thermal_image1.py and capture stdout and stderr
        result = subprocess.run(['python', 'thermal_image1.py'], capture_output=True, text=True)
        
        # Check if the script executed successfully
        if result.returncode == 0:
            # Log and return the script’s output
            print("Script output:", result.stdout)
            return jsonify({'status': 'Fever prediction completed successfully.', 'output': result.stdout})
        else:
            # Log any errors that occurred
            print("Script error:", result.stderr)
            return jsonify({'status': 'Fever prediction failed.', 'error': result.stderr})
    
    except Exception as e:
        # Capture and log any exceptions from running the script
        print("Exception:", str(e))
        return jsonify({'status': 'Error', 'message': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
