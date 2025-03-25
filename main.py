# Import necessary libraries
import cv2  # OpenCV for image processing
import cvzone  # CVZone for simplifying image processing tasks
from cvzone.HandTrackingModule import HandDetector  # Module for hand detection
from picamera2 import Picamera2  # Library for controlling the Picamera2 on Raspberry Pi

# Configure Picamera2
picam2 = Picamera2()  # Create a Picamera2 object
picam2.preview_configuration.main.size = (640, 480)  # Set image resolution (640x480 pixels)
picam2.preview_configuration.main.format = "RGB888"  # Set image format to RGB888
picam2.preview_configuration.align()  # Align image configuration
picam2.configure("preview")  # Set the camera to preview mode
picam2.start()  # Start the camera

# Initialize hand detector
detector = HandDetector(maxHands=2, detectionCon=0.7, minTrackCon=0.7)  # Higher confidence for accuracy

while True:
    # Capture image from the camera (Picamera2 outputs in RGB format)
    im = picam2.capture_array()  
    
    # (No mirroring applied, image remains as it is)
    
    # Detect hands in the image
    hands, im = detector.findHands(im, draw=True)  # Ensure hand landmarks are drawn

    # Initialize counters for raised and bent fingers
    count_1 = 0  # Counter for raised fingers
    count_0 = 0  # Counter for bent fingers

    if hands:
        fingers_list = []  # List to store finger detection results
        for hand in hands:
            fingers = detector.fingersUp(hand)  # Get status of raised (1) and bent (0) fingers
            fingers_list.extend(fingers)  # Append finger status from each hand

        # Count the number of raised and bent fingers
        count_1 = fingers_list.count(1)
        count_0 = fingers_list.count(0)

        # Print the counted values in the console
        print(f"Raised Fingers: {count_1}, Bent Fingers: {count_0}")

    # Display the number of raised fingers in the image
    cvzone.putTextRect(im, f'Fingers Up: {count_1}', (50, 60), scale=2, thickness=2, colorR=(0, 255, 0))

    # Show the processed image in a window
    cv2.imshow("Hand Tracking", im)

    # Exit the loop if ESC key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ASCII code for ESC
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

