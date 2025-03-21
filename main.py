# Import necessary libraries
import cv2  # OpenCV for image processing
import cvzone  # CVZone for simplifying image processing tasks
from cvzone.HandTrackingModule import HandDetector  # Module for hand detection
from picamera2 import Picamera2  # Library for controlling the Picamera2 on Raspberry Pi

# Configure Picamera2
picam2 = Picamera2()  # Create a Picamera2 object
picam2.preview_configuration.main.size = (640, 480)  # Set image resolution (640x480 pixels)
picam2.preview_configuration.main.format = "RGB888"  # Set image format to RGB888
picam2.preview_configuration.align()  # Align image configuration
picam2.configure("preview")  # Set the camera to preview mode
picam2.start()  # Start the camera

# Initialize hand detector
detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)
# maxHands=2 → Detect up to 2 hands
# detectionCon=0.5 → Minimum detection confidence (50%)
# minTrackCon=0.5 → Minimum tracking confidence for detected hands

while True:  # Infinite loop until ESC key is pressed
    # Capture image from the camera
    im = picam2.capture_array()  # Retrieve image from the camera
    im = cv2.flip(im, -1)  # Flip image both vertically and horizontally (if the camera is inverted)

    # Detect hands in the image
    hands, im = detector.findHands(im, draw=True)  
    # `hands` stores detected hand data
    # `draw=True` enables drawing hand outlines in the image

    # Initialize counters for raised and bent fingers
    count_1 = 0  # Counter for raised fingers
    count_0 = 0  # Counter for bent fingers

    if hands:  # If hands are detected
        fingers_list = []  # List to store finger detection results
        for hand in hands:  # Loop through each detected hand
            fingers = detector.fingersUp(hand)  # Get status of raised (1) and bent (0) fingers
            fingers_list.extend(fingers)  # Append finger status from each hand to the list

        # Count the number of raised (1) and bent (0) fingers
        count_1 = fingers_list.count(1)
        count_0 = fingers_list.count(0)

        # Print the counted values in the console
        print("Count of 1s:", count_1)  # Number of raised fingers
        print("Count of 0s:", count_0)  # Number of bent fingers

    # Display the number of raised fingers in the image
    cvzone.putTextRect(im, f'Fingers Up: {count_1}', (50, 60), scale=2, thickness=2)
    # `cvzone.putTextRect()` overlays text on the image with a border
    # `f'Fingers Up: {count_1}'` → Text showing the number of raised fingers
    # `(50, 60)` → Position of text in the image (pixels)
    # `scale=2` → Font size
    # `thickness=2` → Font thickness

    # Display the processed image in a window titled "Hand Tracking"
    cv2.imshow("Hand Tracking", im)

    # Check for key presses
    key = cv2.waitKey(1)  # Wait for key input every 1 millisecond
    if key == 27:  # If ESC key (ASCII code 27) is pressed, exit the loop
        break

# Close all OpenCV windows after pressing ESC
cv2.destroyAllWindows()
