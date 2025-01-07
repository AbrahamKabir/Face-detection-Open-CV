import cv2

# Load the pre-trained Haar cascade classifier for face detection
a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the default camera (webcam)
b = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    c, d_image = b.read()
    
    # Convert the frame to grayscale for face detection
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    f = a.detectMultiScale(e, 1.3, 6)
    
    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 5)
    
    # Display the frame with detected faces
    cv2.imshow('img', d_image)
    
    # Break the loop if the user presses the 'Esc' key
    h = cv2.waitKey(40) & 0xff
    if h == 27:  # ASCII for the 'Esc' key
        break

# Release the camera and close all OpenCV windows
b.release()
cv2.destroyAllWindows()